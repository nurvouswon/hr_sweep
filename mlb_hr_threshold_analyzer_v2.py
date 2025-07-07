import pandas as pd

# ==== LOAD DATA ====
df_event = pd.read_parquet("3_19_7_5.parquet")      # Adjust filename if needed
df_today = pd.read_csv("today_hr_features7_6.csv")  # Adjust filename if needed

print("Event-level shape:", df_event.shape)
print("TODAY shape:", df_today.shape)

# ==== PICK AN ALL-NULL COLUMN TO CHECK ====
TARGET_COL = "b_ff_avg_exit_velo_3"  # Change to any feature that is all-null in TODAY

# ==== CHECK UNIQUE KEYS AND DTYPE ====
print("\nTODAY unique batter_ids:", df_today['batter_id'].nunique())
print("EVENT unique batter_ids:", df_event['batter_id'].nunique())
print("\ndf_event dtypes:")
print(df_event[['game_date', 'batter_id']].dtypes)
print("\ndf_today dtypes:")
print(df_today[['game_date', 'batter_id']].dtypes)

# If dtypes mismatch, fix here (optional):
df_event['batter_id'] = df_event['batter_id'].astype(str)
df_today['batter_id'] = df_today['batter_id'].astype(str)
# (Do this for any other key columns as well)

# ==== CHECK PRESENCE OF TARGET COLUMN DATA ====
print("\nNon-null in EVENT for this column:", df_event[TARGET_COL].notnull().sum())
print("Non-null in TODAY for this column:", df_today[TARGET_COL].notnull().sum())

# Print a few sample batter_ids with values in event-level
print("\nSample EVENT batter_ids with data for", TARGET_COL)
print(df_event.loc[df_event[TARGET_COL].notnull(), ['batter_id', TARGET_COL]].head())

# Print a few sample batter_ids in TODAY
print("\nSample TODAY batter_ids:")
print(df_today[['batter_id', TARGET_COL]].head())

# ==== OPTIONAL: TRY A MERGE TO SEE IF IT WORKS ====
# You may want to simulate the merge/groupby logic that should fill in these features
merge_cols = ['game_date', 'batter_id']  # Adjust if you join on more keys!
sample_cols = [TARGET_COL]  # Or all the feature columns you expect

# Suppose you want to see if a left-merge would join the right data
tmp_event = df_event[merge_cols + sample_cols].drop_duplicates(subset=merge_cols)
test_merge = df_today.merge(tmp_event, on=merge_cols, how='left', suffixes=('', '_event'))

print("\nChecking post-merge, is TARGET_COL_event filled?")
print(test_merge[[TARGET_COL, f"{TARGET_COL}_event"]].head(10))

# ==== UTILITY: FAST CHECKER FUNCTION ====
def debug_null_column(df_event, df_today, col, key='batter_id'):
    print(f"\n=== Debug: {col} ===")
    print("Event-level, non-null:", df_event[col].notnull().sum())
    print("TODAY, non-null:", df_today[col].notnull().sum())
    print("Sample EVENT ids with data:", df_event.loc[df_event[col].notnull(), key].unique()[:5])
    print("Sample TODAY ids:", df_today[key].unique()[:5])
    print("-"*40)

# ==== USAGE: CHECK ANY COLUMN ====
debug_null_column(df_event, df_today, "b_ff_avg_exit_velo_3", key="batter_id")
debug_null_column(df_event, df_today, "b_sl_avg_exit_velo_3", key="batter_id")  # Or any other column

# ==== SUMMARY: WHAT TO LOOK FOR ====
# - If EVENT has data and TODAY is null -> merge/groupby/key mismatch
# - If EVENT is all null, feature logic is broken or not computed
# - If dtypes for keys mismatch, fix with astype(str) or astype(int) as appropriate
# - Use the test_merge block to experiment with different merge keys and confirm join logic
