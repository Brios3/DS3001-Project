# Stage 4: Picks the position-relevant stats and assigns reproducible
# player-level train/validation/test splits with no data leakage.

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# File locations
DATA_DIR = Path(r"C:\Users\bryso\PycharmProjects\PythonProject\Stat 2120\Raw Data")
INPUT_FILE = DATA_DIR / "merged_player_stage3_modeling.csv"
INPUT_FOULS_FILE = DATA_DIR / "merged_player_stage3_fouls_modeling.csv"
INPUT_POSSESSIONS_FILE = DATA_DIR / "merged_player_stage3_possessions_modeling.csv"

OUTPUT_FILE = DATA_DIR / "merged_player_stage4_model_ready.csv"
OUTPUT_FOULS_FILE = DATA_DIR / "merged_player_stage4_fouls_model_ready.csv"
OUTPUT_POSSESSIONS_FILE = DATA_DIR / "merged_player_stage4_possessions_model_ready.csv"
OUTPUT_PLAYERS_FILE = DATA_DIR / "merged_player_stage4_player_splits.csv"
OUTPUT_FEATURES_FILE = DATA_DIR / "merged_player_stage4_features.json"
OUTPUT_REPORT_FILE = DATA_DIR / "merged_player_stage4_summary.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
LOW_MINUTES_THRESHOLD = 120

# Identifiers and the target, not predictors
identifiers = ["NBA_ID", "PLAYER_NAME", "SEASON", "POSITION"]

# Model features: per-12 stats, efficiencies, and context (PF, PFD, and BLKA kept on purpose)
features = [
    "AGE", "MIN_PER_GAME", "GP_TOTAL",
    "PTS_PER12", "FGM_PER12", "FGA_PER12",
    "FG2M_PER12", "FG2A_PER12", "FG3M_PER12", "FG3A_PER12",
    "FTM_PER12", "FTA_PER12",
    "OREB_PER12", "DREB_PER12", "REB_PER12",
    "AST_PER12", "TOV_PER12", "STL_PER12", "BLK_PER12",
    "BLKA_PER12", "PF_PER12", "PFD_PER12",
    "FG_PCT", "FG2_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT",
    "THREE_POINT_ATTEMPT_SHARE", "FREE_THROW_ATTEMPT_RATE",
    "AST_TO_TOV", "OREB_SHARE", "DD2_PER_GAME", "TD3_PER_GAME"
]

possession_features = [
    "TOUCHES_PER12", "FRONT_CT_TOUCHES_PER12", "TIME_OF_POSS_PER12",
    "ELBOW_TOUCHES_PER12", "POST_TOUCHES_PER12", "PAINT_TOUCHES_PER12",
    "AVG_SEC_PER_TOUCH", "AVG_DRIB_PER_TOUCH", "PTS_PER_TOUCH",
    "PTS_PER_ELBOW_TOUCH", "PTS_PER_POST_TOUCH", "PTS_PER_PAINT_TOUCH"
]


def load_dataset(path, cols):
    data = pd.read_csv(path, low_memory=False)

    missing = sorted(set(identifiers + cols).difference(data.columns))
    if missing: raise ValueError(path.name + " is missing columns: " + str(missing))
    if data[identifiers].isna().any().any(): raise ValueError("Missing player ID, name, season, or position in " + path.name + ".")
    if data.duplicated(["NBA_ID", "SEASON"]).any(): raise ValueError("Duplicate player-season records in " + path.name + ".")

    # Reject non-numeric text but allow real missing values
    converted = data[cols].apply(pd.to_numeric, errors="coerce")
    bad = (data[cols].notna() & converted.isna()).any()
    if bad.any(): raise ValueError("Non-numeric values in " + bad.idxmax() + ": " + path.name)
    data[cols] = converted

    if np.isinf(data[cols].to_numpy(dtype=float)).any(): raise ValueError("Infinite predictor values in " + path.name + ".")

    # Stage 3 should have already removed low-minute rows
    if "MIN_TOTAL_APPROX" in data.columns and data["MIN_TOTAL_APPROX"].lt(LOW_MINUTES_THRESHOLD).any():
        raise ValueError("Low-minute records remain in " + path.name + ".")

    return data


# Build the split once, from the largest dataset
main = load_dataset(INPUT_FILE, features)

# Each player needs exactly one position so we can stratify on it
if (main.groupby("NBA_ID")["POSITION"].nunique() > 1).any(): raise ValueError("Players with conflicting positions found. Review labels first.")

player_labels = main.groupby("NBA_ID", as_index=False)["POSITION"].first()

# Split by player, never by season row
train_val_players, test_players = train_test_split(player_labels, test_size=TEST_SIZE, stratify=player_labels["POSITION"], random_state=RANDOM_STATE)
train_players, validation_players = train_test_split(train_val_players, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), stratify=train_val_players["POSITION"], random_state=RANDOM_STATE)

player_splits = pd.concat([train_players.assign(SPLIT="train"), validation_players.assign(SPLIT="validation"), test_players.assign(SPLIT="test")], ignore_index=True)

if player_splits["NBA_ID"].duplicated().any(): raise ValueError("A player was assigned to more than one split.")

player_splits.to_csv(OUTPUT_PLAYERS_FILE, index=False)


def save_model_dataset(data, cols, path):
    # Every feature set uses the same player splits
    result = data[identifiers + cols].merge(player_splits[["NBA_ID", "SPLIT"]], on="NBA_ID", how="left", validate="many_to_one")
    if result["SPLIT"].isna().any(): raise ValueError("Players missing from master split: " + path.name)

    result = result[identifiers + ["SPLIT"] + cols].sort_values(["SPLIT", "SEASON", "NBA_ID"]).reset_index(drop=True)

    # No player can appear in more than one split, and no split can be empty
    if result.groupby("NBA_ID")["SPLIT"].nunique().gt(1).any(): raise ValueError("Player leakage across splits: " + path.name)
    for partition in ["train", "validation", "test"]:
        if not (result["SPLIT"] == partition).any(): raise ValueError("Empty " + partition + " split in " + path.name)

    result.to_csv(path, index=False)
    return result


possession_cols = features + possession_features

main_output = save_model_dataset(main, features, OUTPUT_FILE)
fouls_output = save_model_dataset(load_dataset(INPUT_FOULS_FILE, features), features, OUTPUT_FOULS_FILE)
possessions_output = save_model_dataset(load_dataset(INPUT_POSSESSIONS_FILE, possession_cols), possession_cols, OUTPUT_POSSESSIONS_FILE)

# Record which columns are X and which is y
feature_manifest = {
    "target": "POSITION",
    "identifiers_not_predictors": ["NBA_ID", "PLAYER_NAME", "SEASON"],
    "split_column": "SPLIT",
    "random_state": RANDOM_STATE,
    "split_fractions_by_player": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "main_features": features,
    "fouls_2005plus_features": features,
    "possession_features": possession_cols,
    "missing_value_policy": "Keep NaN here; fit imputer on training split only",
    "class_labels": sorted(main["POSITION"].unique().tolist())
}

OUTPUT_FEATURES_FILE.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

# Short summary report (no modeling happens here)
with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
    print("Stage 4 NBA position modeling preparation", file=f)

    for label, data, cols in [("Main", main_output, features), ("Fouls 2005+", fouls_output, features), ("Possessions", possessions_output, possession_cols)]:
        print(file=f)
        print(label + ": " + format(len(data), ",") + " rows, " + str(len(cols)) + " predictors", file=f)
        print("Player counts by split: " + str(data.groupby("SPLIT")["NBA_ID"].nunique().to_dict()), file=f)
        print("Row counts by split: " + str(data["SPLIT"].value_counts().to_dict()), file=f)
        print("Missing PFD_PER12: " + format(int(data["PFD_PER12"].isna().sum()), ","), file=f)
        print("Class distribution: " + str(data["POSITION"].value_counts().to_dict()), file=f)

print("Stage 4 modeling preparation completed successfully!")
