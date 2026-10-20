# Stage 2: Combines regular-season and playoff stats into comparable
# per-12-minute features, keeping the indicators that help predict position.

import numpy as np
import pandas as pd
from pathlib import Path

# File locations
DATA_DIR = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")
INPUT_FILE = DATA_DIR / "merged_player_stage1_labeled.csv"
OUTPUT_FILE = DATA_DIR / "merged_player_stage2_cleaned.csv"
OUTPUT_FOULS_FILE = DATA_DIR / "merged_player_stage2_fouls_2005plus.csv"
OUTPUT_POSSESSIONS_FILE = DATA_DIR / "merged_player_stage2_possessions.csv"

KEY = ["NBA_ID", "SEASON"]
PFD_CUTOFF_YEAR = 2005
LOW_MINUTES_THRESHOLD = 120

# Load Stage 1 data
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Basic checks on columns, duplicates, and positions
required = ["NBA_ID", "PLAYER_NAME", "SEASON", "POSITION","GP_rs", "GP_po", "MIN_rs", "MIN_po"]

if not set(required).issubset(df.columns):
    raise ValueError("Missing required Stage 1 columns.")

if df.duplicated(KEY).any():
    raise ValueError("Duplicate player-season rows found.")

if df["POSITION"].isna().any():
    raise ValueError("Missing player positions detected.")

# Convert games and minutes to numbers
for suffix in ["rs", "po"]:
    for stat in ["GP", "MIN"]:
        col = f"{stat}_{suffix}"
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].lt(0).any():
            raise ValueError(f"Negative values detected in {col}.")

    # Games played without minutes would break the per-minute math
    if (df[f"GP_{suffix}"].gt(0) & df[f"MIN_{suffix}"].isna()).any():
        raise ValueError(f"Missing minutes for {suffix} games.")

# Total games and approximate total minutes
gp_rs = df["GP_rs"].fillna(0)
gp_po = df["GP_po"].fillna(0)

minutes_total = gp_rs * df["MIN_rs"].fillna(0) + gp_po * df["MIN_po"].fillna(0)
safe_minutes = minutes_total.replace(0, np.nan)


# Small helpers used throughout
def to_num(col):
    return pd.to_numeric(df[col], errors="coerce")


def has_data(rs, po):
    # A stat is usable only if it's present for every part the player actually played
    return (gp_rs.eq(0) | rs.notna()) & (gp_po.eq(0) | po.notna())


def per12(total):
    return total / safe_minutes * 12


def safe_ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


# Start the cleaned dataset with identifying info
cleaned = pd.DataFrame(index=df.index)

cleaned["NBA_ID"] = df["NBA_ID"]
cleaned["PLAYER_NAME"] = df["PLAYER_NAME"]
cleaned["SEASON"] = df["SEASON"]
cleaned["POSITION"] = df["POSITION"]

# Player context and playing time
cleaned["AGE"] = to_num("AGE_rs").combine_first(to_num("AGE_po"))

cleaned["GP_RS"] = gp_rs
cleaned["GP_PO"] = gp_po
cleaned["GP_TOTAL"] = gp_rs + gp_po

cleaned["MIN_TOTAL_APPROX"] = minutes_total
cleaned["MIN_PER_GAME"] = minutes_total / cleaned["GP_TOTAL"].replace(0, np.nan)

# Flag small samples and old seasons with unreliable fouls-drawn data
cleaned["LOW_MINUTES_SAMPLE"] = minutes_total < LOW_MINUTES_THRESHOLD

cleaned["SEASON_START"] = pd.to_numeric(df["SEASON"].str[:4], errors="coerce")
cleaned["PFD_UNRELIABLE_OLD_SEASON"] = cleaned["SEASON_START"] < PFD_CUTOFF_YEAR

# Stats that help predict position
stats = [
    "PTS",
    "FGM", "FGA",
    "FG3M", "FG3A",
    "FTM", "FTA",
    "OREB", "DREB", "REB",
    "AST", "TOV",
    "STL", "BLK", "BLKA",
    "PF", "PFD"
]

# Combine regular season and playoffs into season totals, then per-12 rates
totals = {}

for stat in stats:
    rs = to_num(f"{stat}_rs")
    po = to_num(f"{stat}_po")

    if rs.lt(0).any() or po.lt(0).any():
        raise ValueError(f"Negative values detected for {stat}.")

    # Rebuild totals from per-game averages
    total = (rs.fillna(0) * gp_rs + po.fillna(0) * gp_po).where(has_data(rs, po))

    # Old PFD values stay missing
    if stat == "PFD":
        total = total.mask(cleaned["PFD_UNRELIABLE_OLD_SEASON"])

    totals[stat] = total
    cleaned[f"{stat}_PER12"] = per12(total)

# Two-point makes and attempts (negative results are invalid)
fg2a = totals["FGA"] - totals["FG3A"]
fg2m = totals["FGM"] - totals["FG3M"]

cleaned["FG2A_PER12"] = per12(fg2a.where(fg2a.ge(0)))
cleaned["FG2M_PER12"] = per12(fg2m.where(fg2m.ge(0)))

# Shooting percentages
cleaned["FG_PCT"] = safe_ratio(totals["FGM"], totals["FGA"])
cleaned["FG2_PCT"] = safe_ratio(fg2m, fg2a)
cleaned["FG3_PCT"] = safe_ratio(totals["FG3M"], totals["FG3A"])
cleaned["FT_PCT"] = safe_ratio(totals["FTM"], totals["FTA"])
cleaned["EFG_PCT"] = safe_ratio(totals["FGM"] + 0.5 * totals["FG3M"], totals["FGA"])

# Shooting profile
cleaned["THREE_POINT_ATTEMPT_SHARE"] = safe_ratio(totals["FG3A"], totals["FGA"])
cleaned["FREE_THROW_ATTEMPT_RATE"] = safe_ratio(totals["FTA"], totals["FGA"])

# Playmaking and rebounding
cleaned["AST_TO_TOV"] = safe_ratio(totals["AST"], totals["TOV"])
cleaned["OREB_SHARE"] = safe_ratio(totals["OREB"], totals["REB"])

# Percentages outside 0-1 are impossible, so set them to missing
percentage_cols = [
    "FG_PCT", "FG2_PCT", "FG3_PCT",
    "FT_PCT", "EFG_PCT",
    "THREE_POINT_ATTEMPT_SHARE",
    "OREB_SHARE"
]

for col in percentage_cols:
    cleaned[col] = cleaned[col].where(cleaned[col].between(0, 1) | cleaned[col].isna())

# Double-doubles and triple-doubles are already counts, so just add them up
for stat in ["DD2", "TD3"]:
    rs = to_num(f"{stat}_rs")
    po = to_num(f"{stat}_po")

    total = (rs.fillna(0) + po.fillna(0)).where(has_data(rs, po))
    cleaned[f"{stat}_PER_GAME"] = safe_ratio(total, cleaned["GP_TOTAL"])

# Save the main dataset
cleaned = cleaned.sort_values(["SEASON", "NBA_ID"]).reset_index(drop=True)
cleaned.to_csv(OUTPUT_FILE, index=False)

# Save a separate dataset where fouls-drawn data is reliable
fouls = cleaned[cleaned["SEASON_START"] >= PFD_CUTOFF_YEAR].copy()
fouls.to_csv(OUTPUT_FOULS_FILE, index=False)

# Possession stats (only available for some seasons)
possession_counts = [
    "TOUCHES",
    "FRONT_CT_TOUCHES",
    "TIME_OF_POSS",
    "ELBOW_TOUCHES",
    "POST_TOUCHES",
    "PAINT_TOUCHES"
]

possession_averages = [
    "AVG_SEC_PER_TOUCH",
    "AVG_DRIB_PER_TOUCH",
    "PTS_PER_TOUCH",
    "PTS_PER_ELBOW_TOUCH",
    "PTS_PER_POST_TOUCH",
    "PTS_PER_PAINT_TOUCH"
]

numeric_possession_cols = ["MIN"] + possession_counts + possession_averages

possessions = df[KEY + numeric_possession_cols].copy()
possessions[numeric_possession_cols] = possessions[numeric_possession_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Keep rows that have minutes and touch data
possessions = possessions[
    possessions["MIN"].gt(0) & possessions["TOUCHES"].notna()
].copy()

# Counts become per-12-minute rates
for stat in possession_counts:
    possessions[f"{stat}_PER12"] = possessions[stat] / possessions["MIN"] * 12

keep_possessions = (
    KEY
    + [f"{stat}_PER12" for stat in possession_counts]
    + possession_averages
)

# Join with the main stats and save
modern = cleaned.merge(
    possessions[keep_possessions],
    on=KEY,
    how="inner",
    validate="one_to_one"
)

modern.to_csv(OUTPUT_POSSESSIONS_FILE, index=False)

print("Stage 2 NBA cleaning completed successfully!")
