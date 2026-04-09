"""
P1: Match Result Prediction - EPL Project
Goal: Predict Home Win / Draw / Away Win (3-class classification)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR  = BASE_DIR / "models" / "p1_match_result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("P1: Match Result Prediction")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading data...")
mr  = pd.read_parquet(DATA_DIR / "match_results.parquet")
tss = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")
pml = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")

mr = mr.sort_values("MatchDate").reset_index(drop=True)
print(f"  match_results      : {mr.shape}")
print(f"  team_season_summary: {tss.shape}")
print(f"  player_match_logs  : {pml.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2] Engineering features...")

# ---------- 2a. ELO Ratings ----------
ELO_K    = 20
ELO_INIT = 1500

def expected_elo(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

elo_ratings = defaultdict(lambda: ELO_INIT)
home_elo_list, away_elo_list = [], []

for _, row in mr.iterrows():
    h, a = row["HomeTeam"], row["AwayTeam"]
    re   = row["FullTimeResult"]

    rh = elo_ratings[h]
    ra = elo_ratings[a]
    home_elo_list.append(rh)
    away_elo_list.append(ra)

    exp_h = expected_elo(rh, ra)
    exp_a = 1.0 - exp_h

    if re == "H":
        sh, sa = 1.0, 0.0
    elif re == "A":
        sh, sa = 0.0, 1.0
    else:
        sh, sa = 0.5, 0.5

    elo_ratings[h] = rh + ELO_K * (sh - exp_h)
    elo_ratings[a] = ra + ELO_K * (sa - exp_a)

mr["home_elo_before"] = home_elo_list
mr["away_elo_before"] = away_elo_list
mr["elo_diff"]        = mr["home_elo_before"] - mr["away_elo_before"]
print("  ELO ratings done")

# ---------- 2b. Rolling form features ----------
ROLL_N = 5

# Build a long-form team match history (one row per team per match)
# Points: Win=3, Draw=1, Loss=0
def get_result_for_team(row, team):
    if row["HomeTeam"] == team:
        if row["FullTimeResult"] == "H": return 3
        elif row["FullTimeResult"] == "D": return 1
        else: return 0
    else:
        if row["FullTimeResult"] == "A": return 3
        elif row["FullTimeResult"] == "D": return 1
        else: return 0

def get_gf_ga_for_team(row, team):
    if row["HomeTeam"] == team:
        return row["FullTimeHomeGoals"], row["FullTimeAwayGoals"]
    else:
        return row["FullTimeAwayGoals"], row["FullTimeHomeGoals"]

# Build per-team sorted history
team_history = defaultdict(list)
for idx, row in mr.iterrows():
    h, a = row["HomeTeam"], row["AwayTeam"]
    pts_h = get_result_for_team(row, h)
    pts_a = get_result_for_team(row, a)
    gf_h, ga_h = get_gf_ga_for_team(row, h)
    gf_a, ga_a = get_gf_ga_for_team(row, a)
    team_history[h].append((row["MatchDate"], idx, pts_h, gf_h, ga_h, "home"))
    team_history[a].append((row["MatchDate"], idx, pts_a, gf_a, ga_a, "away"))

# For each match row, compute rolling stats BEFORE this match
# We need: avg pts last 5, avg gf last 5, avg ga last 5, home-specific form, away-specific form
roll_features = {
    "home_roll_pts"     : np.nan * np.ones(len(mr)),
    "away_roll_pts"     : np.nan * np.ones(len(mr)),
    "home_roll_gf"      : np.nan * np.ones(len(mr)),
    "home_roll_ga"      : np.nan * np.ones(len(mr)),
    "away_roll_gf"      : np.nan * np.ones(len(mr)),
    "away_roll_ga"      : np.nan * np.ones(len(mr)),
    "home_home_roll_pts": np.nan * np.ones(len(mr)),
    "away_away_roll_pts": np.nan * np.ones(len(mr)),
    "home_days_since"   : np.nan * np.ones(len(mr)),
    "away_days_since"   : np.nan * np.ones(len(mr)),
}

# Sort each team's history by date
for team in team_history:
    team_history[team].sort(key=lambda x: x[0])

# Build index: for each team, sorted list of (date, match_idx, ...)
for idx, row in mr.iterrows():
    match_date = row["MatchDate"]
    for role, team in [("home", row["HomeTeam"]), ("away", row["AwayTeam"])]:
        hist = team_history[team]
        # matches BEFORE current
        past = [e for e in hist if e[1] != idx and e[0] <= match_date]
        past_sorted = sorted(past, key=lambda x: x[0])

        last_n = past_sorted[-ROLL_N:] if len(past_sorted) >= 1 else []
        last_n_home = [e for e in past_sorted if e[5] == "home"][-ROLL_N:]
        last_n_away = [e for e in past_sorted if e[5] == "away"][-ROLL_N:]

        if last_n:
            pts_vals = [e[2] for e in last_n]
            gf_vals  = [e[3] for e in last_n]
            ga_vals  = [e[4] for e in last_n]
            roll_features[f"{role}_roll_pts"][idx]  = np.mean(pts_vals)
            roll_features[f"{role}_roll_gf"][idx]   = np.mean(gf_vals)
            roll_features[f"{role}_roll_ga"][idx]   = np.mean(ga_vals)
            # days since last match
            last_date = past_sorted[-1][0]
            roll_features[f"{role}_days_since"][idx] = (match_date - last_date).days

        if role == "home" and last_n_home:
            roll_features["home_home_roll_pts"][idx] = np.mean([e[2] for e in last_n_home])
        elif role == "away" and last_n_away:
            roll_features["away_away_roll_pts"][idx] = np.mean([e[2] for e in last_n_away])

for col, vals in roll_features.items():
    mr[col] = vals

print("  Rolling form features done")

# ---------- 2c. Head-to-head last 5 ----------
h2h_home_pts = np.nan * np.ones(len(mr))
h2h_away_pts = np.nan * np.ones(len(mr))

for idx, row in mr.iterrows():
    h, a = row["HomeTeam"], row["AwayTeam"]
    match_date = row["MatchDate"]
    # past meetings between these two teams (either direction)
    mask = (
        ((mr["HomeTeam"] == h) & (mr["AwayTeam"] == a) |
         (mr["HomeTeam"] == a) & (mr["AwayTeam"] == h)) &
        (mr["MatchDate"] < match_date)
    )
    past_h2h = mr[mask].tail(ROLL_N)
    if len(past_h2h) > 0:
        home_pts_list = []
        away_pts_list = []
        for _, pr in past_h2h.iterrows():
            home_pts_list.append(get_result_for_team(pr, h))
            away_pts_list.append(get_result_for_team(pr, a))
        h2h_home_pts[idx] = np.mean(home_pts_list)
        h2h_away_pts[idx] = np.mean(away_pts_list)

mr["h2h_home_pts"] = h2h_home_pts
mr["h2h_away_pts"] = h2h_away_pts
print("  Head-to-head features done")

# ---------- 2d. Season stage ----------
def season_stage(row):
    season = row["Season"]
    date   = row["MatchDate"]
    season_rows = mr[mr["Season"] == season]
    if len(season_rows) < 3:
        return 1
    s_min = season_rows["MatchDate"].min()
    s_max = season_rows["MatchDate"].max()
    total_days = (s_max - s_min).days
    if total_days == 0:
        return 1
    elapsed = (date - s_min).days / total_days
    if elapsed < 0.33:
        return 0   # early
    elif elapsed < 0.66:
        return 1   # mid
    else:
        return 2   # late

mr["season_stage"] = mr.apply(season_stage, axis=1)
print("  Season stage done")

# ---------- 2e. Goal difference momentum ----------
# Difference: avg goals scored - avg goals conceded (last 5)
mr["home_gd_momentum"] = mr["home_roll_gf"] - mr["home_roll_ga"]
mr["away_gd_momentum"] = mr["away_roll_gf"] - mr["away_roll_ga"]

# ---------- 2f. Merge season-level team stats ----------
# Use previous season's stats as features
tss_prev = tss.copy()
season_list = sorted(tss_prev["Season"].unique())
season_order = {s: i for i, s in enumerate(season_list)}
tss_prev["season_idx"] = tss_prev["Season"].map(season_order)
tss_prev_shifted = tss_prev.copy()
tss_prev_shifted["season_idx"] = tss_prev_shifted["season_idx"] + 1
tss_prev_shifted = tss_prev_shifted.rename(columns={
    "points":            "prev_season_pts",
    "goal_diff":         "prev_season_gd",
    "total_wins":        "prev_season_wins",
    "home_wins":         "prev_season_home_wins",
    "away_wins":         "prev_season_away_wins",
})
idx_to_season = {v: k for k, v in season_order.items()}
tss_prev_shifted["Season"] = tss_prev_shifted["season_idx"].map(idx_to_season)

merge_cols = ["Season", "team", "prev_season_pts", "prev_season_gd",
              "prev_season_wins", "prev_season_home_wins", "prev_season_away_wins"]
tss_merge = tss_prev_shifted[merge_cols].dropna(subset=["Season"])

mr = mr.merge(
    tss_merge.rename(columns={"team": "HomeTeam"}).add_prefix("home_").rename(
        columns={"home_Season": "Season", "home_HomeTeam": "HomeTeam"}),
    on=["Season", "HomeTeam"], how="left"
)
mr = mr.merge(
    tss_merge.rename(columns={"team": "AwayTeam"}).add_prefix("away_").rename(
        columns={"away_Season": "Season", "away_AwayTeam": "AwayTeam"}),
    on=["Season", "AwayTeam"], how="left"
)
print("  Season-level team stats merged")

# ─────────────────────────────────────────────
# 3. TARGET & FEATURE SELECTION
# ─────────────────────────────────────────────
label_map = {"H": 0, "D": 1, "A": 2}
mr["target"] = mr["FullTimeResult"].map(label_map)

FEATURE_COLS = [
    # ELO
    "home_elo_before", "away_elo_before", "elo_diff",
    # Rolling form
    "home_roll_pts", "away_roll_pts",
    "home_roll_gf", "home_roll_ga",
    "away_roll_gf", "away_roll_ga",
    "home_home_roll_pts", "away_away_roll_pts",
    # H2H
    "h2h_home_pts", "h2h_away_pts",
    # Days since last match
    "home_days_since", "away_days_since",
    # Season stage
    "season_stage",
    # Momentum
    "home_gd_momentum", "away_gd_momentum",
    # Previous season stats
    "home_prev_season_pts", "home_prev_season_gd",
    "home_prev_season_wins", "home_prev_season_home_wins",
    "away_prev_season_pts", "away_prev_season_gd",
    "away_prev_season_wins", "away_prev_season_away_wins",
]

# ─────────────────────────────────────────────
# 4. TIME-BASED DATA SPLIT
# ─────────────────────────────────────────────
TRAIN_SEASONS = [s for s in mr["Season"].unique() if s <= "2020/21"]
VAL_SEASONS   = [s for s in mr["Season"].unique() if "2021/22" <= s <= "2022/23"]
TEST_SEASONS  = [s for s in mr["Season"].unique() if s >= "2023/24"]

train_df = mr[mr["Season"].isin(TRAIN_SEASONS)].copy()
val_df   = mr[mr["Season"].isin(VAL_SEASONS)].copy()
test_df  = mr[mr["Season"].isin(TEST_SEASONS)].copy()

print(f"\n[3] Data split:")
print(f"  Train : {len(train_df)} matches  ({train_df['Season'].min()} – {train_df['Season'].max()})")
print(f"  Val   : {len(val_df)} matches  ({val_df['Season'].min()} – {val_df['Season'].max()})")
print(f"  Test  : {len(test_df)} matches  ({test_df['Season'].min()} – {test_df['Season'].max()})")

def split_xy(df):
    X = df[FEATURE_COLS].copy()
    y = df["target"].values
    return X, y

X_train, y_train = split_xy(train_df)
X_val,   y_val   = split_xy(val_df)
X_test,  y_test  = split_xy(test_df)

# Impute missing with median from train
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_val_sc   = scaler.transform(X_val_imp)
X_test_sc  = scaler.transform(X_test_imp)

# Class weights for imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
cw_dict = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
print(f"\n  Class weights: H={cw_dict[0]:.3f}, D={cw_dict[1]:.3f}, A={cw_dict[2]:.3f}")

# ─────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)

def evaluate(name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro")
    cm   = confusion_matrix(y_true, y_pred).tolist()
    prec, rec, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2])
    report = classification_report(y_true, y_pred,
                                   target_names=["HomeWin","Draw","AwayWin"],
                                   digits=4)
    print(f"\n  --- {name} ---")
    print(f"  Accuracy : {acc:.4f}   |   F1-macro : {f1m:.4f}")
    print(report)
    result = {
        "accuracy"       : round(float(acc), 4),
        "f1_macro"       : round(float(f1m), 4),
        "confusion_matrix": cm,
        "per_class": {
            "HomeWin": {"precision": round(float(prec[0]),4),
                        "recall":    round(float(rec[0]),4),
                        "f1":        round(float(f1_cls[0]),4)},
            "Draw":    {"precision": round(float(prec[1]),4),
                        "recall":    round(float(rec[1]),4),
                        "f1":        round(float(f1_cls[1]),4)},
            "AwayWin": {"precision": round(float(prec[2]),4),
                        "recall":    round(float(rec[2]),4),
                        "f1":        round(float(f1_cls[2]),4)},
        }
    }
    return result

all_results = {}

# ─────────────────────────────────────────────
# 5a. MODEL 1 – XGBoost
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4a] Training XGBoost...")
print("=" * 60)

import xgboost as xgb
import pickle

sample_weight = np.array([cw_dict[y] for y in y_train])

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)

xgb_model.fit(
    X_train_imp, y_train,
    sample_weight=sample_weight,
    eval_set=[(X_val_imp, y_val)],
    verbose=50,
)

y_pred_xgb_val  = xgb_model.predict(X_val_imp)
y_pred_xgb_test = xgb_model.predict(X_test_imp)

print("\n  Validation:")
val_xgb  = evaluate("XGBoost - Validation", y_val,  y_pred_xgb_val)
print("\n  Test:")
test_xgb = evaluate("XGBoost - Test",       y_test, y_pred_xgb_test)

# Feature importance
fi = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)
fi.to_csv(OUT_DIR / "xgb_feature_importance.csv", index=False)
print(f"\n  Top-10 features:\n{fi.head(10).to_string(index=False)}")

# Save model
with open(OUT_DIR / "xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("  Saved: xgb_model.pkl")

all_results["xgboost"] = {"validation": val_xgb, "test": test_xgb}

# ─────────────────────────────────────────────
# 5b. MODEL 2 – LSTM/GRU (PyTorch)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4b] Training LSTM/GRU (PyTorch)...")
print("=" * 60)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10   # last N matches per team
N_FEAT  = len(FEATURE_COLS)
print(f"  Device: {DEVICE}")

# ── Build sequences: for each match, stack the last SEQ_LEN feature vectors
# We use the imputed + scaled feature array aligned to mr rows
mr_feat_scaled = np.full((len(mr), N_FEAT), np.nan)
mr_feat_scaled[train_df.index] = X_train_sc
mr_feat_scaled[val_df.index]   = X_val_sc
mr_feat_scaled[test_df.index]  = X_test_sc

# For rows not in any split (shouldn't happen) just impute 0
mr_feat_scaled = np.nan_to_num(mr_feat_scaled, nan=0.0)

# Build per-team ordered match index list
team_match_idx = defaultdict(list)
for idx, row in mr.iterrows():
    team_match_idx[row["HomeTeam"]].append(idx)
    team_match_idx[row["AwayTeam"]].append(idx)

def build_sequence_dataset(df_split):
    """Return (X_seq [N, SEQ_LEN, FEAT], y [N])"""
    seqs, labels = [], []
    for idx, row in df_split.iterrows():
        h, a    = row["HomeTeam"], row["AwayTeam"]
        cur_pos = mr.index.get_loc(idx)
        # Get last SEQ_LEN rows BEFORE current match for home team
        h_hist = [i for i in team_match_idx[h] if i < idx]
        a_hist = [i for i in team_match_idx[a] if i < idx]
        h_last = h_hist[-SEQ_LEN:] if len(h_hist) >= 1 else []
        a_last = a_hist[-SEQ_LEN:] if len(a_hist) >= 1 else []
        # Pad to SEQ_LEN
        h_seq = np.zeros((SEQ_LEN, N_FEAT))
        a_seq = np.zeros((SEQ_LEN, N_FEAT))
        if h_last:
            h_arr = mr_feat_scaled[h_last]
            h_seq[-len(h_arr):] = h_arr
        if a_last:
            a_arr = mr_feat_scaled[a_last]
            a_seq[-len(a_arr):] = a_arr
        # Concatenate home + away sequences → (SEQ_LEN, 2*N_FEAT)
        seq = np.concatenate([h_seq, a_seq], axis=1)
        seqs.append(seq)
        labels.append(row["target"])
    X_seq = np.array(seqs, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.int64)
    return X_seq, y_seq

print("  Building sequence datasets...")
X_seq_train, y_seq_train = build_sequence_dataset(train_df)
X_seq_val,   y_seq_val   = build_sequence_dataset(val_df)
X_seq_test,  y_seq_test  = build_sequence_dataset(test_df)
print(f"  Sequence shape: train={X_seq_train.shape}, val={X_seq_val.shape}, test={X_seq_test.shape}")

class MatchSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = MatchSeqDataset(X_seq_train, y_seq_train)
val_ds   = MatchSeqDataset(X_seq_val,   y_seq_val)
test_ds  = MatchSeqDataset(X_seq_test,  y_seq_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# Class weights tensor
cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn  = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)           # (B, T, H)
        out    = out[:, -1, :]         # last timestep
        out    = self.bn(out)
        out    = self.drop(out)
        return self.fc(out)

lstm_model = GRUClassifier(
    input_size  = 2 * N_FEAT,
    hidden_size = 128,
    num_layers  = 2,
    dropout     = 0.3,
    num_classes = 3,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=cw_tensor)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=False
)

EPOCHS      = 60
best_val_f1 = -1.0
best_state  = None

print("  Training GRU model...")
for epoch in range(1, EPOCHS + 1):
    lstm_model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = lstm_model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    avg_loss = total_loss / len(train_ds)

    # Validate
    lstm_model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = lstm_model(xb)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())
    val_f1 = f1_score(all_true, all_preds, average="macro")
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | val_f1={val_f1:.4f} (best={best_val_f1:.4f})")

# Load best
lstm_model.load_state_dict(best_state)

def predict_loader(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())
    return np.array(preds), np.array(trues)

y_pred_gru_val,  y_true_gru_val  = predict_loader(lstm_model, val_loader)
y_pred_gru_test, y_true_gru_test = predict_loader(lstm_model, test_loader)

print("\n  Validation:")
val_gru  = evaluate("GRU - Validation", y_true_gru_val,  y_pred_gru_val)
print("\n  Test:")
test_gru = evaluate("GRU - Test",       y_true_gru_test, y_pred_gru_test)

torch.save(lstm_model.state_dict(), OUT_DIR / "gru_model.pt")
print("  Saved: gru_model.pt")

all_results["gru"] = {"validation": val_gru, "test": test_gru}

# ─────────────────────────────────────────────
# 5c. MODEL 3 – TabNet / MLP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[4c] Training TabNet / MLP (PyTorch)...")
print("=" * 60)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    USE_TABNET = True
    print("  Using TabNet")
except ImportError:
    USE_TABNET = False
    print("  TabNet not available — using MLP instead")

if USE_TABNET:
    tabnet_model = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.3,
        n_independent=2, n_shared=2,
        momentum=0.02,
        mask_type="sparsemax",
        verbose=10,
        seed=42,
        device_name=str(DEVICE),
    )
    sample_weight_arr = np.array([class_weights[y] for y in y_train])
    tabnet_model.fit(
        X_train=X_train_imp.astype(np.float32),
        y_train=y_train,
        eval_set=[(X_val_imp.astype(np.float32), y_val)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=200,
        patience=20,
        batch_size=512,
        virtual_batch_size=256,
        weights=1,          # auto class weights
    )
    y_pred_tab_val  = tabnet_model.predict(X_val_imp.astype(np.float32))
    y_pred_tab_test = tabnet_model.predict(X_test_imp.astype(np.float32))

    tabnet_model.save_model(str(OUT_DIR / "tabnet_model"))
    print("  Saved: tabnet_model")
    model3_name = "tabnet"

else:
    # MLP in PyTorch
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3, num_classes=3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev_dim = h
            layers.append(nn.Linear(prev_dim, num_classes))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class TabularDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    tab_train_ds = TabularDataset(X_train_sc, y_train)
    tab_val_ds   = TabularDataset(X_val_sc,   y_val)
    tab_test_ds  = TabularDataset(X_test_sc,  y_test)
    tab_train_ld = DataLoader(tab_train_ds, batch_size=256, shuffle=True)
    tab_val_ld   = DataLoader(tab_val_ds,   batch_size=512, shuffle=False)
    tab_test_ld  = DataLoader(tab_test_ds,  batch_size=512, shuffle=False)

    mlp_model = MLP(
        input_dim   = N_FEAT,
        hidden_dims = (256, 128, 64),
        dropout     = 0.3,
        num_classes = 3,
    ).to(DEVICE)

    opt_mlp  = torch.optim.Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_m  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_mlp, mode="max", factor=0.5, patience=8, verbose=False
    )
    crit_mlp = nn.CrossEntropyLoss(weight=cw_tensor)

    MLP_EPOCHS      = 100
    best_mlp_f1     = -1.0
    best_mlp_state  = None

    print("  Training MLP model...")
    for epoch in range(1, MLP_EPOCHS + 1):
        mlp_model.train()
        tot_loss = 0.0
        for xb, yb in tab_train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_mlp.zero_grad()
            loss = crit_mlp(mlp_model(xb), yb)
            loss.backward()
            opt_mlp.step()
            tot_loss += loss.item() * len(yb)

        mlp_model.eval()
        preds_v, trues_v = [], []
        with torch.no_grad():
            for xb, yb in tab_val_ld:
                xb = xb.to(DEVICE)
                preds_v.extend(mlp_model(xb).argmax(1).cpu().numpy())
                trues_v.extend(yb.numpy())
        vf1 = f1_score(trues_v, preds_v, average="macro")
        sched_m.step(vf1)

        if vf1 > best_mlp_f1:
            best_mlp_f1    = vf1
            best_mlp_state = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}

        if epoch % 20 == 0:
            avg_loss = tot_loss / len(tab_train_ds)
            print(f"  Epoch {epoch:3d}/{MLP_EPOCHS} | loss={avg_loss:.4f} | val_f1={vf1:.4f} (best={best_mlp_f1:.4f})")

    mlp_model.load_state_dict(best_mlp_state)

    def pred_tab_loader(model, loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                preds.extend(model(xb).argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        return np.array(preds), np.array(trues)

    y_pred_tab_val,  y_true_tab_val  = pred_tab_loader(mlp_model, tab_val_ld)
    y_pred_tab_test, y_true_tab_test = pred_tab_loader(mlp_model, tab_test_ld)

    # Override true labels from loaders (same order)
    y_val_for_tab  = y_true_tab_val
    y_test_for_tab = y_true_tab_test

    torch.save(mlp_model.state_dict(), OUT_DIR / "mlp_model.pt")
    print("  Saved: mlp_model.pt")
    model3_name = "mlp"

# Evaluate model 3
if USE_TABNET:
    y_val_for_tab  = y_val
    y_test_for_tab = y_test

print("\n  Validation:")
val_tab  = evaluate(f"{model3_name.upper()} - Validation", y_val_for_tab,  y_pred_tab_val)
print("\n  Test:")
test_tab = evaluate(f"{model3_name.upper()} - Test",       y_test_for_tab, y_pred_tab_test)

all_results[model3_name] = {"validation": val_tab, "test": test_tab}

# ─────────────────────────────────────────────
# 6. COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[5] Final Comparison Table")
print("=" * 60)
print(f"\n{'Model':<12} {'Val Acc':>8} {'Val F1':>8} {'Test Acc':>9} {'Test F1':>9}")
print("-" * 50)
for model_name, metrics in all_results.items():
    v_acc = metrics["validation"]["accuracy"]
    v_f1  = metrics["validation"]["f1_macro"]
    t_acc = metrics["test"]["accuracy"]
    t_f1  = metrics["test"]["f1_macro"]
    print(f"{model_name:<12} {v_acc:>8.4f} {v_f1:>8.4f} {t_acc:>9.4f} {t_f1:>9.4f}")

# ─────────────────────────────────────────────
# 7. SAVE RESULTS
# ─────────────────────────────────────────────
results_path = OUT_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved results to: {results_path}")

# Save feature importance
fi_dict = dict(zip(FEATURE_COLS, xgb_model.feature_importances_.tolist()))
all_results["xgb_feature_importance"] = fi_dict
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

# Save imputer & scaler for inference
with open(OUT_DIR / "imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
with open(OUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(OUT_DIR / "feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print(f"\n  Saved artifacts to: {OUT_DIR}")
print("\n" + "=" * 60)
print("P1 Training Complete!")
print("=" * 60)
