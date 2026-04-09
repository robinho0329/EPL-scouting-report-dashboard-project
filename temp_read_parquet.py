import pandas as pd

# File 1: transfer_targets
print('='*80)
print('FILE 1: s2_v4_2025_transfer_targets.parquet')
print('='*80)
df1 = pd.read_parquet(r'C:\Users\xcv54\workspace\EPL project\data\scout\s2_v4_2025_transfer_targets.parquet')
print(f'Shape: {df1.shape}')
print(f'Columns: {list(df1.columns)}')
print('First 15 rows:')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 30)
print(df1.head(15).to_string())

# File 2: undervalued
print()
print('='*80)
print('FILE 2: s2_v4_undervalued.parquet')
print('='*80)
df2 = pd.read_parquet(r'C:\Users\xcv54\workspace\EPL project\data\scout\s2_v4_undervalued.parquet')
print(f'Shape: {df2.shape}')
if 'season' in df2.columns:
    print('season value_counts:')
    print(df2['season'].value_counts().to_string())
else:
    print('No season column. Columns:', list(df2.columns))

# File 3: all_predictions
print()
print('='*80)
print('FILE 3: s2_v4_all_predictions.parquet')
print('='*80)
df3 = pd.read_parquet(r'C:\Users\xcv54\workspace\EPL project\data\scout\s2_v4_all_predictions.parquet')
print(f'Shape: {df3.shape}')
if 'season' in df3.columns:
    print('season value_counts:')
    print(df3['season'].value_counts().to_string())
else:
    print('No season column. Columns:', list(df3.columns))
