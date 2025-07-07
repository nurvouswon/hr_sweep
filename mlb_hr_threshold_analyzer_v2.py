import streamlit as st
import pandas as pd

st.title("MLB HR Feature Debugger & Safe Merger")

# --- Uploaders ---
event_file = st.file_uploader("Upload Event-Level Parquet", type="parquet")
today_file = st.file_uploader("Upload TODAY CSV", type="csv")

if event_file and today_file:
    df_event = pd.read_parquet(event_file)
    df_today = pd.read_csv(today_file)

    st.write(f"Event-level shape: {df_event.shape}")
    st.write(f"TODAY shape: {df_today.shape}")

    # --- Diagnose duplicate keys ---
    merge_keys = ['game_date', 'batter_id']
    event_dups = df_event.duplicated(subset=merge_keys, keep=False)
    today_dups = df_today.duplicated(subset=merge_keys, keep=False)

    st.write("Event-level duplicates on [game_date, batter_id]:", int(event_dups.sum()))
    st.write("TODAY duplicates on [game_date, batter_id]:", int(today_dups.sum()))

    if event_dups.any():
        st.warning("Duplicate rows in EVENT detected for (game_date, batter_id)! Aggregating...")
        st.write(df_event.loc[event_dups, merge_keys].head(10))
        # --- Aggregate: Take first row per key. (Customize this as needed for your use-case!)
        df_event_grouped = df_event.groupby(merge_keys).first().reset_index()
        st.success(f"Aggregated event-level shape: {df_event_grouped.shape}")
    else:
        df_event_grouped = df_event

    # --- Show columns present in TODAY but not in Event-level, and vice versa ---
    cols_today = set(df_today.columns)
    cols_event = set(df_event_grouped.columns)
    only_in_today = sorted(cols_today - cols_event)
    only_in_event = sorted(cols_event - cols_today)
    in_both = sorted(cols_today & cols_event)

    st.write("Columns in TODAY but NOT in Event-level:", only_in_today)
    st.write("Columns in Event-level but NOT in TODAY:", only_in_event)
    st.write("Columns in BOTH (sample):", in_both[:10])

    # --- Merge and report nulls ---
    merged = df_today.merge(
        df_event_grouped, on=merge_keys, how='left', suffixes=('', '_event')
    )
    st.write("Merged shape:", merged.shape)

    # Null diagnostics
    nulls = merged.isnull().sum()
    all_null_cols = nulls[nulls == merged.shape[0]].index.tolist()
    st.write("Columns ALL NULL after merge:", all_null_cols[:20])
    st.write(f"Total columns all-null: {len(all_null_cols)}")

    st.write("Sample merged rows:")
    st.dataframe(merged.head())

    # Download merged result
    st.download_button(
        "Download merged CSV", merged.to_csv(index=False).encode(),
        file_name="merged_today_event.csv", mime="text/csv"
    )
else:
    st.info("Please upload both Event-level and TODAY files.")
