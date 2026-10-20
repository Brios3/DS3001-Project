# Stage 1 cleaning for the merged NBA dataset, before combining
# regular-season and playoff stats for position modeling.

import pandas as pd
import numpy as np
from pathlib import Path

# File locations
DATA_DIR = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")
INPUT_FILE = DATA_DIR / "merged_player.csv"
OUTPUT_FILE = DATA_DIR / "merged_player_stage1_cleaned.csv"
OUTPUT_LABELED_FILE = DATA_DIR / "merged_player_stage1_labeled.csv"

KEY = ["NBA_ID", "SEASON"]
PFD_CUTOFF_YEAR = 2005
LOW_MINUTES_THRESHOLD = 120

# Load data and clean up text columns
df = pd.read_csv(INPUT_FILE, low_memory=False)
original_rows = len(df)

df["PLAYER_NAME"] = df["PLAYER_NAME"].astype("string").str.strip()
df["SEASON"] = df["SEASON"].astype("string").str.strip()
df["POSITION"] = df["POSITION"].astype("string").str.strip().replace("", pd.NA)

# One player ID from whichever source column has it
id_cols = ["PLAYER_ID_rs", "PLAYER_ID_po", "PLAYER_ID"]
ids = df[id_cols].apply(pd.to_numeric, errors="coerce")

if ids.nunique(axis=1).gt(1).any():
    raise ValueError("Conflicting NBA player IDs detected.")

df["NBA_ID"] = ids.bfill(axis=1).iloc[:, 0].astype("Int64")

if df["NBA_ID"].isna().any() or df["SEASON"].isna().any():
    raise ValueError("Missing player IDs or season information.")

# Make sure duplicate player-seasons don't contradict each other
duplicates = df[df.duplicated(KEY, keep=False)]

if not duplicates.empty:
    distinct_counts = duplicates.groupby(KEY).nunique().drop(columns="PLAYER_NAME")

    if distinct_counts["POSITION"].gt(1).any():
        raise ValueError("Conflicting position labels.")

    conflicts = distinct_counts.drop(columns="POSITION").sort_index(axis=1).gt(1).any()
    if conflicts.any():
        raise ValueError(f"Conflicting values found in {conflicts.idxmax()}.")

# Merge duplicates into one row, preferring regular-season data
df["_SOURCE_PRIORITY"] = (
    df["GP_rs"].notna().astype(int) * 2
    + df["GP_po"].notna().astype(int)
)

df = df.sort_values("_SOURCE_PRIORITY", ascending=False, kind="stable")
df = df.groupby(KEY, as_index=False, sort=False).first()
df = df.drop(columns="_SOURCE_PRIORITY")

# Fill missing positions if the player has only one position in other seasons
by_player = df.groupby("NBA_ID")["POSITION"]
position_lookup = by_player.first().where(by_player.nunique() == 1)
lookup_values = df["NBA_ID"].map(position_lookup)

df["POSITION_RECOVERED"] = df["POSITION"].isna() & lookup_values.notna()
df["POSITION"] = df["POSITION"].fillna(lookup_values)

# Convert games, minutes, and fouls to numbers
numeric_cols = [
    f"{stat}_{suffix}"
    for suffix in ["rs", "po"]
    for stat in ["GP", "MIN", "PF", "PFD"]
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

has_negative = df[numeric_cols].lt(0).any()
if has_negative.any():
    raise ValueError(f"Negative values found in {has_negative.idxmax()}.")

# Fouls drawn is unreliable in old seasons, so set it to missing (not zero)
df["SEASON_START"] = pd.to_numeric(df["SEASON"].str[:4], errors="coerce")

if df["SEASON_START"].isna().any():
    raise ValueError("Invalid season labels detected.")

df["PFD_UNRELIABLE_OLD_SEASON"] = df["SEASON_START"] < PFD_CUTOFF_YEAR
df.loc[df["PFD_UNRELIABLE_OLD_SEASON"], ["PFD_rs", "PFD_po"]] = np.nan

# Total minutes across regular season and playoffs
df["TOTAL_MINUTES_EST"] = (
    df["GP_rs"].fillna(0) * df["MIN_rs"].fillna(0)
    + df["GP_po"].fillna(0) * df["MIN_po"].fillna(0)
)

# Flags (small samples are flagged, not deleted)
df["HAS_TRADITIONAL_STATS"] = df["TOTAL_MINUTES_EST"] > 0
df["HAS_POSSESSION_STATS"] = pd.to_numeric(df["TOUCHES"], errors="coerce").notna()
df["LOW_MINUTES_SAMPLE"] = (
    df["HAS_TRADITIONAL_STATS"]
    & df["TOTAL_MINUTES_EST"].lt(LOW_MINUTES_THRESHOLD)
)

# Sort and save
df = df.sort_values(["SEASON", "NBA_ID"]).reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False)

# Labeled set: known position and real minutes played
labeled = df[df["POSITION"].notna() & df["HAS_TRADITIONAL_STATS"]].copy()
labeled.to_csv(OUTPUT_LABELED_FILE, index=False)

print("Stage 1 NBA data cleaning completed successfully!")
print("Original observations:", original_rows)
print("Cleaned observations:", len(df))
print("Labeled observations:", len(labeled))
