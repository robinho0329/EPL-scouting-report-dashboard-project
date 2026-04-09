"""
P1: Match Result Prediction Pipeline
=====================================
3-class classification: Home Win (H) / Draw (D) / Away Win (A)

Models: XGBoost, LSTM, MLP
Data split: Train (2000-2021), Val (2021-2023), Test (2023-2025)
"""

import os
import json
import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\xcv54\workspace\EPL project")
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "match_features.parquet"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "match_results.parquet"
OUTPUT_DIR = PROJECT_ROOT / "models" / "p1_match_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Load & Prepare Data ────────────────────────────────────────────────────
def load_data():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PATH)
    print(f"  match_features shape: {df.shape}")

    # Target: FullTimeResult -> numeric labels
    label_enc = LabelEncoder()
    df["target"] = label_enc.fit_transform(df["FullTimeResult"])  # A=0, D=1, H=2
    print(f"  Classes: {dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))}")
    print(f"  Distribution: {df['FullTimeResult'].value_counts().to_dict()}")

    # Feature columns: all numeric except target-leaking columns
    # Exclude: match outcome columns (goals, result, halftime, shots, cards, fouls, corners)
    # These are known AFTER the match, so they leak information
    exclude_cols = [
        "Season", "MatchDate", "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
        "HalfTimeHomeGoals", "HalfTimeAwayGoals", "HalfTimeResult",
        "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
        "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls",
        "HomeYellowCards", "AwayYellowCards", "HomeRedCards", "AwayRedCards",
        "season_data_missing", "own_goal_flag_home", "own_goal_flag_away", "own_goal_flag",
        "target", "data_split",
    ]
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Split by data_split column
    train_df = df[df["data_split"] == "train"].copy()
    val_df = df[df["data_split"] == "val"].copy()
    test_df = df[df["data_split"] == "test"].copy()
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Extract features & targets
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    # Handle missing values with median imputation (from train set)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save imputer, scaler, label_encoder, feature_cols for later use
    with open(OUTPUT_DIR / "preprocessing.pkl", "wb") as f:
        pickle.dump({
            "imputer": imputer,
            "scaler": scaler,
            "label_encoder": label_enc,
            "feature_cols": feature_cols,
        }, f)
    print("  Saved preprocessing.pkl")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_val_scaled": X_val_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "label_enc": label_enc, "feature_cols": feature_cols,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


# ── 2. Evaluation Helper ──────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label_enc, split_name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  [{split_name}] Accuracy: {acc:.4f}  |  F1 (macro): {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=label_enc.classes_))
    return {"accuracy": round(acc, 4), "f1_macro": round(f1, 4), "confusion_matrix": cm.tolist()}


# ── 3. XGBoost ─────────────────────────────────────────────────────────────────
def train_xgboost(data):
    print("\n" + "=" * 70)
    print("MODEL 1: XGBoost")
    print("=" * 70)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=SEED,
        use_label_encoder=False,
        verbosity=0,
    )

    model.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    results = {}
    for split, X, y in [
        ("val", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]:
        preds = model.predict(X)
        results[split] = evaluate(y, preds, data["label_enc"], split)

    # Also report train performance
    train_preds = model.predict(data["X_train"])
    results["train"] = evaluate(data["y_train"], train_preds, data["label_enc"], "train")

    # Save model
    model_path = OUTPUT_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved {model_path}")

    # Feature importance (top 15)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\n  Top 15 features:")
    for i in top_idx:
        print(f"    {data['feature_cols'][i]:30s} {importances[i]:.4f}")

    return results


# ── 4. MLP (PyTorch) ───────────────────────────────────────────────────────────
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), num_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(data):
    print("\n" + "=" * 70)
    print("MODEL 2: MLP (PyTorch)")
    print("=" * 70)

    input_dim = data["X_train_scaled"].shape[1]
    model = MLPModel(input_dim).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # DataLoaders
    def make_loader(X, y, batch_size=256, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(data["X_train_scaled"], data["y_train"])
    val_loader = make_loader(data["X_val_scaled"], data["y_val"], shuffle=False)
    test_loader = make_loader(data["X_test_scaled"], data["y_test"], shuffle=False)

    # Class weights for imbalanced data
    class_counts = np.bincount(data["y_train"])
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_f1 = 0
    patience_counter = 0
    max_patience = 25
    epochs = 200

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                out = model(X_batch)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_true.extend(y_batch.numpy())

        val_f1 = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_f1)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "mlp_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_DIR / "mlp_model.pt"))
    model.eval()
    print(f"  Best Val F1: {best_val_f1:.4f}")

    # Evaluate
    results = {}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(DEVICE)
                out = model(X_batch)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_true.extend(y_batch.numpy())
        results[split] = evaluate(np.array(all_true), np.array(all_preds), data["label_enc"], split)

    # Train eval
    train_loader_eval = make_loader(data["X_train_scaled"], data["y_train"], shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in train_loader_eval:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y_batch.numpy())
    results["train"] = evaluate(np.array(all_true), np.array(all_preds), data["label_enc"], "train")

    print(f"  Saved mlp_model.pt")
    return results


# ── 5. LSTM (PyTorch) ──────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state
        return self.fc(hn[-1])


def build_sequences(df, feature_cols, scaler, imputer, seq_len=5):
    """
    For each match, build a sequence of the last `seq_len` matches for the home team.
    This creates temporal context for the LSTM.
    """
    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate").reset_index(drop=True)

    # Get numeric features (already imputed and scaled outside)
    X_all = scaler.transform(imputer.transform(df[feature_cols].values.astype(np.float32)))
    y_all = LabelEncoder().fit(["A", "D", "H"]).transform(df["FullTimeResult"].values)

    # Build team-level match history for sequences
    # For each match, we take the last seq_len matches involving the home team
    sequences = []
    targets = []
    indices = []

    # Create a mapping: for each team, track their match indices in order
    team_history = {}  # team -> list of row indices

    for idx in range(len(df)):
        home = df.iloc[idx]["HomeTeam"]
        away = df.iloc[idx]["AwayTeam"]

        # Get home team's last seq_len matches
        home_hist = team_history.get(home, [])

        if len(home_hist) >= seq_len:
            seq = np.array([X_all[i] for i in home_hist[-seq_len:]])
            sequences.append(seq)
            targets.append(y_all[idx])
            indices.append(idx)

        # Update history: both teams played this match
        for team in [home, away]:
            if team not in team_history:
                team_history[team] = []
            team_history[team].append(idx)

    return np.array(sequences), np.array(targets), indices


def train_lstm(data):
    print("\n" + "=" * 70)
    print("MODEL 3: LSTM (PyTorch)")
    print("=" * 70)

    # Reload full df to build sequences properly
    df = pd.read_parquet(FEATURES_PATH)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate").reset_index(drop=True)

    # Load preprocessing
    with open(OUTPUT_DIR / "preprocessing.pkl", "rb") as f:
        preproc = pickle.load(f)

    feature_cols = preproc["feature_cols"]
    imputer = preproc["imputer"]
    scaler = preproc["scaler"]
    label_enc = preproc["label_encoder"]

    seq_len = 5
    print(f"  Building sequences (seq_len={seq_len})...")

    X_seq, y_seq, seq_indices = build_sequences(df, feature_cols, scaler, imputer, seq_len)
    print(f"  Total sequences: {X_seq.shape[0]}, shape: {X_seq.shape}")

    # Split sequences by original data_split
    split_col = df["data_split"].values
    train_mask = np.array([split_col[i] == "train" for i in seq_indices])
    val_mask = np.array([split_col[i] == "val" for i in seq_indices])
    test_mask = np.array([split_col[i] == "test" for i in seq_indices])

    X_train_seq, y_train_seq = X_seq[train_mask], y_seq[train_mask]
    X_val_seq, y_val_seq = X_seq[val_mask], y_seq[val_mask]
    X_test_seq, y_test_seq = X_seq[test_mask], y_seq[test_mask]
    print(f"  Train: {len(X_train_seq)}, Val: {len(X_val_seq)}, Test: {len(X_test_seq)}")

    input_dim = X_train_seq.shape[2]
    model = LSTMModel(input_dim).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def make_loader(X, y, batch_size=256, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train_seq, y_train_seq)
    val_loader = make_loader(X_val_seq, y_val_seq, shuffle=False)
    test_loader = make_loader(X_test_seq, y_test_seq, shuffle=False)

    # Class weights
    class_counts = np.bincount(y_train_seq)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_f1 = 0
    patience_counter = 0
    max_patience = 25
    epochs = 200

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                out = model(X_batch)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_true.extend(y_batch.numpy())

        val_f1 = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_f1)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "lstm_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(OUTPUT_DIR / "lstm_model.pt"))
    model.eval()
    print(f"  Best Val F1: {best_val_f1:.4f}")

    results = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(DEVICE)
                out = model(X_batch)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_true.extend(y_batch.numpy())
        # Rebuild loader without shuffle for train eval
        if split == "train":
            loader_eval = make_loader(X_train_seq, y_train_seq, shuffle=False)
            all_preds, all_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in loader_eval:
                    X_batch = X_batch.to(DEVICE)
                    out = model(X_batch)
                    all_preds.extend(out.argmax(1).cpu().numpy())
                    all_true.extend(y_batch.numpy())
        results[split] = evaluate(np.array(all_true), np.array(all_preds), label_enc, split)

    print(f"  Saved lstm_model.pt")
    return results


# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    data = load_data()

    all_results = {}

    # XGBoost
    all_results["xgboost"] = train_xgboost(data)

    # MLP
    all_results["mlp"] = train_mlp(data)

    # LSTM
    all_results["lstm"] = train_lstm(data)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        "task": "P1 Match Result Prediction (3-class: H/D/A)",
        "features_file": str(FEATURES_PATH),
        "n_features": len(data["feature_cols"]),
        "feature_columns": data["feature_cols"],
        "splits": {
            "train": int(len(data["y_train"])),
            "val": int(len(data["y_val"])),
            "test": int(len(data["y_test"])),
        },
        "class_mapping": {c: int(i) for c, i in zip(
            data["label_enc"].classes_,
            data["label_enc"].transform(data["label_enc"].classes_)
        )},
        "models": {},
    }

    for model_name, res in all_results.items():
        summary["models"][model_name] = {}
        for split in ["train", "val", "test"]:
            if split in res:
                summary["models"][model_name][split] = {
                    "accuracy": res[split]["accuracy"],
                    "f1_macro": res[split]["f1_macro"],
                    "confusion_matrix": res[split]["confusion_matrix"],
                }

    # Print comparison table
    print(f"\n{'Model':<12} {'Val Acc':>8} {'Val F1':>8} {'Test Acc':>9} {'Test F1':>8}")
    print("-" * 50)
    for model_name in ["xgboost", "mlp", "lstm"]:
        res = all_results[model_name]
        va = res.get("val", {}).get("accuracy", 0)
        vf = res.get("val", {}).get("f1_macro", 0)
        ta = res.get("test", {}).get("accuracy", 0)
        tf = res.get("test", {}).get("f1_macro", 0)
        print(f"{model_name:<12} {va:>8.4f} {vf:>8.4f} {ta:>9.4f} {tf:>8.4f}")

    # Save summary
    summary_path = OUTPUT_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results summary to {summary_path}")

    # List all saved files
    print("\nSaved files:")
    for fp in sorted(OUTPUT_DIR.glob("*")):
        if fp.is_file():
            size_mb = fp.stat().st_size / 1024 / 1024
            print(f"  {fp.name:30s} {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
