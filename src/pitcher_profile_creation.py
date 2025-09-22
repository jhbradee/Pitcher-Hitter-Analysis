import os
import glob
import pandas as pd
import pybaseball
import numpy as np


def load_raw_parquet_files(raw_folder="data/raw"):
    """
    Loads all parquet files from the specified directory into a list of DataFrames.
    Returns a list of (filename, DataFrame) tuples.
    """
    parquet_files = glob.glob(os.path.join(raw_folder, "*.parquet"))
    season_dataframes = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        season_dataframes.append((file, df))
        print(f"Loaded {file} with shape {df.shape}")
    return season_dataframes


# load all raw data files and combine into a single DataFrame
all_raw = load_raw_parquet_files()
all_dfs = [df for _, df in all_raw]
raw_df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined raw DataFrame shape: {raw_df.shape}")

# previewing the first few rows of the combined DataFrame and column names for reference
# print("\nCombined Raw DataFrame Preview:")
# print(list(raw_df.columns))
# print(raw_df.head())

# create enhanced pitch-level dataset with all contextual information
pitch_level_df = raw_df.copy()

# add derived features to each pitch
pitch_level_df["in_zone"] = (
    (pitch_level_df["plate_x"].abs() <= 0.83)
    & (pitch_level_df["plate_z"] >= pitch_level_df["sz_bot"])
    & (pitch_level_df["plate_z"] <= pitch_level_df["sz_top"])
)

# result flags
pitch_level_df["is_strike"] = pitch_level_df["description"].isin(
    ["called_strike", "swinging_strike", "foul"]
)
pitch_level_df["is_whiff"] = pitch_level_df["description"] == "swinging_strike"
pitch_level_df["is_contact"] = pitch_level_df["description"].isin(
    ["hit_into_play", "foul", "foul_tip", "hit_into_play_no_out"]
)
pitch_level_df["is_called_strike"] = pitch_level_df["description"] == "called_strike"
pitch_level_df["is_chase"] = (~pitch_level_df["in_zone"]) & (
    pitch_level_df["description"].isin(
        ["swinging_strike", "foul", "hit_into_play", "hit_into_play_no_out"]
    )
)
pitch_level_df["is_barrel"] = (pitch_level_df["launch_speed"] >= 98.0) & (
    pitch_level_df["launch_angle"].between(26, 30)
)

# count context
pitch_level_df["count_state"] = (
    pitch_level_df["balls"].astype(str) + "-" + pitch_level_df["strikes"].astype(str)
)
pitch_level_df["is_ahead_count"] = pitch_level_df["strikes"] > pitch_level_df["balls"]
pitch_level_df["is_behind_count"] = pitch_level_df["balls"] > pitch_level_df["strikes"]
pitch_level_df["is_even_count"] = pitch_level_df["balls"] == pitch_level_df["strikes"]

# game situation
pitch_level_df["runners_on"] = (
    pitch_level_df["on_1b"].notna().astype(int)
    + pitch_level_df["on_2b"].notna().astype(int)
    + pitch_level_df["on_3b"].notna().astype(int)
)
pitch_level_df["risp"] = pitch_level_df["on_2b"] | pitch_level_df["on_3b"].notna()

# pressure situations
pitch_level_df["high_leverage"] = (
    (pitch_level_df["inning"] >= 7)
    & (pitch_level_df["runners_on"] > 0)
    & (abs(pitch_level_df["post_home_score"] - pitch_level_df["post_away_score"]) <= 3)
)

# add pitch sequence context (previous pitch)
pitch_level_df = pitch_level_df.sort_values(
    ["game_pk", "at_bat_number", "pitch_number"]
)
pitch_level_df["prev_pitch_type"] = pitch_level_df.groupby(
    ["game_pk", "at_bat_number"]
)["pitch_type"].shift(1)
pitch_level_df["prev_pitch_result"] = pitch_level_df.groupby(
    ["game_pk", "at_bat_number"]
)["description"].shift(1)
pitch_level_df["prev_strike"] = pitch_level_df.groupby(["game_pk", "at_bat_number"])[
    "is_strike"
].shift(1)

print(f"\nPitch-Level DataFrame shape: {pitch_level_df.shape}")

# save comprehensive pitch-level dataset
pitch_level_df.to_parquet(
    "data/processed/pitch_level_enhanced.parquet",
    index=False,
)
print(
    "\nEnhanced pitch-level data saved to 'data/processed/pitch_level_enhanced.parquet'"
)

###################################
# create aggregated player profiles
###################################

# filter for at-bat ending events for aggregation
at_bat_ending_events = [
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "strikeout",
    "force_out",
    "fielders_choice_out",
    "grounded_into_double_play",
]

filtered_df = pitch_level_df[pitch_level_df["events"].isin(at_bat_ending_events)].copy()
filtered_df.dropna(subset=["pitcher", "batter"], inplace=True)

# pitcher profile by ptich type
pitcher_by_pitch_type_df = (
    filtered_df.groupby(["pitcher", "player_name", "pitch_type", "pitch_name"])
    .agg(
        {
            # volume
            "pitch_type": "count",
            # velocity and movement
            "release_speed": ["mean", "std"],
            "release_spin_rate": ["mean", "std"],
            "pfx_x": "mean",
            "pfx_z": "mean",
            # location
            "plate_x": ["mean", "std"],
            "plate_z": ["mean", "std"],
            "in_zone": "mean",
            # results
            "is_whiff": "mean",
            "is_called_strike": "mean",
            "is_chase": "mean",
            "launch_speed": "mean",
            "launch_angle": "mean",
            "estimated_woba_using_speedangle": "mean",
            "woba_value": "mean",
            "is_barrel": "mean",
            # situational
            "is_ahead_count": "mean",
            "is_behind_count": "mean",
            "risp": "mean",
            "high_leverage": "mean",
        }
    )
    .round(3)
)

# flatten column names
pitcher_by_pitch_type_df.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0]
    for col in pitcher_by_pitch_type_df.columns
]
pitcher_by_pitch_type_df.rename(
    columns={"pitch_type_count": "total_pitches"}, inplace=True
)
pitcher_by_pitch_type_df = pitcher_by_pitch_type_df.reset_index()

# calculate pitch type percentages
total_pitch_by_type = pitcher_by_pitch_type_df.groupby("pitcher")[
    "total_pitches"
].transform("sum")
pitcher_by_pitch_type_df["pitch_type_percentage"] = (
    pitcher_by_pitch_type_df["total_pitches"] / total_pitch_by_type * 100
).round(2)

# add CSW rate and other derived metrics
pitcher_by_pitch_type_df["csw_rate"] = (
    pitcher_by_pitch_type_df["is_called_strike_mean"]
    + pitcher_by_pitch_type_df["is_whiff_mean"]
).round(3)
pitcher_by_pitch_type_df["command_score"] = (
    (pitcher_by_pitch_type_df["in_zone_mean"] * 0.6)
    + (pitcher_by_pitch_type_df["is_called_strike_mean"] * 0.4)
).round(3)
pitcher_by_pitch_type_df["stuff_score"] = (
    (pitcher_by_pitch_type_df["is_whiff_mean"] * 0.7)
    + (pitcher_by_pitch_type_df["release_speed_mean"] / 100 * 0.3)
).round(3)

print(f"\nPitcher by Pitch Type DataFrame shape: {pitcher_by_pitch_type_df.shape}")

# save pitch-type specific profiles
pitcher_by_pitch_type_df.to_parquet(
    "data/processed/pitcher_profile_by_pitch_type.parquet",
    index=False,
)
print(
    "\nPitcher profile by pitch type data saved to 'data/processed/pitcher_profile_by_pitch_type.parquet'"
)

#################################
# Overall pitcher summary (aggregated across all pitches)
#################################

pitcher_overall_df = (
    filtered_df.groupby(["pitcher", "player_name"])
    .agg(
        {
            # volume
            "pitch_type": "count",
            "pitch_name": "nunique",
            # overall performance
            "is_whiff": "mean",
            "is_strike": "mean",
            "in_zone": "mean",
            "woba_value": "mean",
            "launch_speed": "mean",
            "estimated_woba_using_speedangle": "mean",
            "is_barrel": "mean",
            # velocity (across all pitch types)
            "release_speed": "mean",
            # situational
            "is_ahead_count": "mean",
            "is_behind_count": "mean",
            "risp": "mean",
            "high_leverage": "mean",
            # platoon
            "stand": lambda x: (x == "L").mean(),  # percentage faced vs lefties
        }
    )
    .round(3)
)

pitcher_overall_df.columns = [
    "total_pitches",
    "repertoire_size",
    "overall_whiff_rate",
    "overall_strike_rate",
    "overall_zone_rate",
    "overall_woba_against",
    "avg_launch_speed_against",
    "overall_xwoba_against",
    "barrel_rate_against",
    "avg_velocity",
    "pct_ahead_in_count",
    "pct_behind_in_count",
    "pct_risp",
    "pct_high_leverage",
    "pct_vs_lefties",
]

pitcher_overall_df = pitcher_overall_df.reset_index()

# add usage-weighted repertoire stats from pitch-type data
repertoire_stats = (
    pitcher_by_pitch_type_df.groupby("pitcher")
    .agg(
        {
            "total_pitches": "sum",
            "csw_rate": lambda x: np.average(
                x,
                weights=pitcher_by_pitch_type_df.loc[x.index, "pitch_type_percentage"],
            ),
            "stuff_score": lambda x: np.average(
                x,
                weights=pitcher_by_pitch_type_df.loc[x.index, "pitch_type_percentage"],
            ),
            "command_score": lambda x: np.average(
                x,
                weights=pitcher_by_pitch_type_df.loc[x.index, "pitch_type_percentage"],
            ),
        }
    )
    .round(3)
)

repertoire_stats.columns = [
    "total_pitches_check",
    "usage_weighted_csw",
    "usage_weighted_stuff",
    "usage_weighted_command",
]
repertoire_stats = repertoire_stats.reset_index()

# merge with overall pitcher stats
pitcher_overall_df = pitcher_overall_df.merge(
    repertoire_stats[
        [
            "pitcher",
            "usage_weighted_csw",
            "usage_weighted_stuff",
            "usage_weighted_command",
        ]
    ],
    on="pitcher",
    how="left",
)

print(f"\nOverall Pitcher Summary DataFrame shape: {pitcher_overall_df.shape}")

# save overall pitcher summary
pitcher_overall_df.to_parquet(
    "data/processed/pitcher_overall_summary.parquet",
    index=False,
)
print(
    "\nOverall pitcher summary data saved to 'data/processed/pitcher_overall_summary.parquet'"
)

#################################
# Summary
#################################
print(f"\n{'=' * 60}")
print("DATA STRUCTURE SUMMARY")
print(f"{'=' * 60}")
print(
    f"1. Pitch-level data: {pitch_level_df.shape[0]:,} rows, {pitch_level_df.shape[1]} columns"
)
print("   - Every pitch with full context and situational data")
print("   - File: pitch_level_enhanced.parquet")
print()
print(
    f"2. Pitcher by Pitch Type: {pitcher_by_pitch_type_df.shape[0]:,} rows, {pitcher_by_pitch_type_df.shape[1]} columns"
)
print("   - Aggregated metrics per pitcher per pitch type")
print("   - File: pitcher_profile_by_pitch_type.parquet")
print()
print(
    f"3. Overall Pitcher Summary: {pitcher_overall_df.shape[0]:,} rows, {pitcher_overall_df.shape[1]} columns"
)
print("   - Aggregated metrics per pitcher across all pitch types")
print("   - File: pitcher_overall_summary.parquet")
print(f"{'=' * 60}\n")
