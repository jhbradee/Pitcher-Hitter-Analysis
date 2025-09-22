import os
import glob
import pandas as pd
import pybaseball
import numpy as np

# assuming pitch_level_df is already laoded from the pitcher profile creation script

####################################################
# Batter Profile Creation
####################################################

# filter for at-bat ending events for batter analysis
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

# use the pitch_level_df from pitcher profile creation
if "pitch_level_df" not in locals():
    pitch_level_df = pd.read_parquet("data/processed/pitch_level_enhanced.parquet")

filtered_df = pitch_level_df[pitch_level_df["events"].isin(at_bat_ending_events)].copy()
filtered_df.dropna(subset=["pitcher", "batter"], inplace=True)

# get batter names
unique_batters = filtered_df["batter"].unique().tolist()
batter_names = pybaseball.playerid_reverse_lookup(unique_batters, key_type="mlbam")
batter_names["formatted_name"] = (
    batter_names["name_last"].str.capitalize()
    + ", "
    + batter_names["name_first"].str.capitalize()
)
batter_name_lookup = batter_names[["formatted_name", "key_mlbam"]].rename(
    columns={"key_mlbam": "batter"}
)

# add additional batter-specific features to filtered data
filtered_df["is_competitive_ab"] = ~filtered_df["events"].isin(
    ["walk"]
)  # exclude walks
filtered_df["is_power_contact"] = (filtered_df["launch_speed"] >= 95) & (
    filtered_df["launch_angle"].between(15, 35)
)
filtered_df["is_hard_contact"] = filtered_df["launch_speed"] >= 95
filtered_df["is_soft_contact"] = (filtered_df["launch_speed"] <= 80) & (
    filtered_df["launch_speed"] > 0
)
filtered_df["is_ground_ball"] = (filtered_df["launch_angle"] <= 10) & (
    filtered_df["launch_angle"] >= -90
)
filtered_df["is_line_drive"] = filtered_df["launch_angle"].between(10, 25)
filtered_df["is_fly_ball"] = filtered_df["launch_angle"] > 25
filtered_df["is_popup"] = filtered_df["launch_angle"] > 50

# chase and discipline metrics
filtered_df["swung_at"] = filtered_df["description"].isin(
    ["swinging_strike", "foul_tip", "foul", "hit_into_play"]
)
filtered_df["swung_and_missed"] = filtered_df["description"] == "swinging_strike"
filtered_df["made_contact"] = filtered_df["description"].isin(
    ["foul_tip", "foul", "hit_into_play"]
)

####################################################
# Batter by Pitch Type Profiles
####################################################

batter_by_pitch_type_df = (
    filtered_df.groupby(["batter", "pitch_type", "pitch_name"])
    .agg(
        {
            # volume
            "pitch_type": "count",
            # contact and swing metrics
            "swung_at": "mean",
            "swung_and_missed": "mean",
            "made_contact": "mean",
            "is_whiff": "mean",
            "is_contact": "mean",
            # chase and discipline
            "is_chase": "mean",
            "in_zone": lambda x: filtered_df.loc[x.index, "swung_at"][
                filtered_df.loc[x.index, "in_zone"]
            ].mean()
            if filtered_df.loc[x.index, "in_zone"].any()
            else np.nan,  # swing rate in zone
            # quality of contact
            "launch_speed": ["mean", "std"],
            "launch_angle": ["mean", "std"],
            "is_barrel": "mean",
            "is_hard_contact": "mean",
            "is_power_contact": "mean",
            "is_soft_contact": "mean",
            # batted ball types
            "is_ground_ball": "mean",
            "is_line_drive": "mean",
            "is_fly_ball": "mean",
            "is_popup": "mean",
            # expected outcomes
            "estimated_ba_using_speedangle": "mean",
            "estimated_woba_using_speedangle": "mean",
            "woba_value": "mean",
            # situational performance
            "is_ahead_count": "mean",
            "is_behind_count": "mean",
            "risp": "mean",
            "high_leverage": "mean",
            # speed seen
            "release_speed": "mean",
        }
    )
    .round(3)
)

# flatten column names
batter_by_pitch_type_df.columns = [
    "_".join([str(x) for x in col if x])
    for col in batter_by_pitch_type_df.columns.to_flat_index()
]
batter_by_pitch_type_df.rename(
    columns={"pitch_type_count": "pitches_seen"}, inplace=True
)
batter_by_pitch_type_df.reset_index()

# add batter names
batter_by_pitch_type_df = batter_by_pitch_type_df.merge(
    batter_name_lookup, on="batter", how="left"
)

# calculate percentage of pitches seen by type for each batter
total_pitches_seen = batter_by_pitch_type_df.groupby("batter")[
    "pitches_seen"
].transform("sum")
batter_by_pitch_type_df["pitch_type_seen_pct"] = (
    batter_by_pitch_type_df["pitches_seen"] / total_pitches_seen * 100
).round(2)

# add derived metrics
batter_by_pitch_type_df["contact_rate"] = (
    1 - batter_by_pitch_type_df["is_whiff_mean"]
).round(3)
batter_by_pitch_type_df["swing_rate"] = batter_by_pitch_type_df["swung_at_mean"].round(
    3
)
batter_by_pitch_type_df["whiff_rate"] = batter_by_pitch_type_df["is_whiff_mean"].round(
    3
)
batter_by_pitch_type_df["chase_rate"] = (
    batter_by_pitch_type_df["is_chase_mean"]
).round(3)

# quality metrics
batter_by_pitch_type_df["barrel_rate"] = batter_by_pitch_type_df[
    "is_barrel_mean"
].round(3)
batter_by_pitch_type_df["hard_contact_rate"] = batter_by_pitch_type_df[
    "is_hard_contact_mean"
].round(3)
batter_by_pitch_type_df["avg_exit_velocity"] = batter_by_pitch_type_df[
    "launch_speed_mean"
].round(1)
batter_by_pitch_type_df["avg_launch_angle"] = batter_by_pitch_type_df[
    "launch_angle_mean"
].round(1)

print(f"\nBatter by Pitch Type DataFrame shape: {batter_by_pitch_type_df.shape}")

# save pitch-type specific batter profiles
batter_by_pitch_type_df.to_parquet(
    "data/processed/batter_by_pitch_type_profiles.parquet", index=False
)

print(
    "Batter by Pitch Type Profiles saved to data/processed/batter_by_pitch_type_profiles.parquet"
)

####################################################
# Batter vs Pitcher Handedness
####################################################

batter_vs_handedness_df = (
    filtered_df.groupby(["batter", "p_throws"])
    .agg(
        {
            # volume
            "pitch_type": "count",
            # performance metrics
            "woba_value": "mean",
            "estimated_woba_using_speedangle": "mean",
            "launch_speed": "mean",
            "launch_angle": "mean",
            "is_barrel": "mean",
            "is_hard_contact": "mean",
            "is_power_contact": "mean",
            # discipline
            "swung_at": "mean",
            "is_whiff": "mean",
            "is_chase": "mean",
            # batted ball profile
            "is_ground_ball": "mean",
            "is_line_drive": "mean",
            "is_fly_ball": "mean",
        }
    )
    .round(3)
)

# flatten column names
batter_vs_handedness_df.columns = [
    "_".join([str(x) for x in col if x])
    for col in batter_vs_handedness_df.columns.to_flat_index()
]
batter_vs_handedness_df.rename(
    columns={"pitch_type_count": "pitches_faced"}, inplace=True
)
batter_vs_handedness_df = batter_vs_handedness_df.reset_index()

# pivot to get vs LHP and RHP in seperate columns
handedness_pivot = batter_vs_handedness_df.pivot_table(
    index="batter",
    columns="p_throws",
    values=[
        col
        for col in batter_vs_handedness_df.columns
        if col not in ["batter", "p_throws"]
    ],
    fill_value=np.nan,
)

# flatten multi-level columns with full stat name and handedness (e.g., 'is_hard_contact_mean_L')
handedness_pivot.columns = [
    f"{'_'.join([str(x) for x in col if x])}"
    for col in handedness_pivot.columns.to_flat_index()
]
handedness_pivot = handedness_pivot.reset_index()

print(f"\nBatter vs Pitcher Handedness DataFrame shape: {handedness_pivot.shape}")

# save batter vs handedness profiles
handedness_pivot.to_parquet(
    "data/processed/batter_vs_pitcher_handedness.parquet", index=False
)
print(
    "Batter vs Pitcher Handedness Profiles saved to data/processed/batter_vs_pitcher_handedness.parquet"
)

####################################################
# count-specific performance
####################################################

batter_by_count_df = (
    filtered_df.groupby(["batter", "balls", "strikes"])
    .agg(
        {
            # volume
            "pitch_type": "count",
            # performance metrics
            "woba_value": "mean",
            "launch_speed": "mean",
            # discipline
            "swung_at": "mean",
            "is_whiff": "mean",
            "is_chase": "mean",
            # batted ball profile
            "in_zone": "mean",
        }
    )
    .round(3)
)

batter_by_count_df.columns = [
    "pitches_seen",
    "woba",
    "avg_exit_velocity",
    "swing_rate",
    "whiff_rate",
    "chase_rate",
    "zone_rate_seen",
]
batter_by_count_df = batter_by_count_df.reset_index()

# focus on key counts
key_counts = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (0, 2),
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 1),
    (3, 0),
    (2, 2),
    (3, 2),
]
count_features = pd.DataFrame()

for balls, strikes in key_counts:
    count_subset_df = batter_by_count_df[
        (batter_by_count_df["balls"] == balls)
        & (batter_by_count_df["strikes"] == strikes)
    ]
    count_subset_df = count_subset_df.drop(columns=["balls", "strikes"])
    count_subset_df = count_subset_df[
        ["batter", "woba", "swing_rate", "whiff_rate"]
    ].copy()
    count_subset_df.columns = [
        "batter",
        f"woba_{balls}_{strikes}",
        f"swing_rate_{balls}_{strikes}",
        f"whiff_rate_{balls}_{strikes}",
    ]

    if count_features.empty:
        count_features = count_subset_df
    else:
        count_features = count_features.merge(count_subset_df, on="batter", how="outer")

print(f"\nBatter Count features DataFrame shape: {count_features.shape}")

################################################
# Overall Batter Summary (aggregated across all situations)
################################################

batter_overall_df = (
    filtered_df.groupby("batter")
    .agg(
        {
            # volume
            "pitch_type": "count",
            # performance metrics
            "woba_value": "mean",
            "estimated_woba_using_speedangle": "mean",
            "estimated_ba_using_speedangle": "mean",
            # contact quality
            "launch_speed": ["mean", "std"],
            "launch_angle": ["mean", "std"],
            "is_barrel": "mean",
            "is_hard_contact": "mean",
            "is_power_contact": "mean",
            "is_soft_contact": "mean",
            # discipline
            "swung_at": "mean",
            "is_whiff": "mean",
            "is_chase": "mean",
            "in_zone": lambda x: filtered_df.loc[x.index, "swung_at"][
                filtered_df.loc[x.index, "in_zone"]
            ].mean(),  # zone swing rate
            # batted ball profile
            "is_ground_ball": "mean",
            "is_line_drive": "mean",
            "is_fly_ball": "mean",
            "is_popup": "mean",
            # situational performance
            "risp": "mean",
            "high_leverage": "mean",
            "is_ahead_count": "mean",
            "is_behind_count": "mean",
            # speed seen
            "release_speed": "mean",
            # handedness faced
            "p_throws": lambda x: (x == "L").mean(),  # percentage vs lefties
            "stand": "first",
        }
    )
    .round(3)
)

batter_overall_df.columns = [
    "total_pitches_seen",
    "overall_woba",
    "overall_xwoba",
    "overall_xba",
    "avg_exit_velocity",
    "exit_velocity_std",
    "avg_launch_angle",
    "launch_angle_std",
    "barrel_rate",
    "hard_contact_rate",
    "power_contact_rate",
    "soft_contact_rate",
    "swing_rate",
    "overall_whiff_rate",
    "chase_rate",
    "zone_swing_rate",
    "ground_ball_rate",
    "line_drive_rate",
    "fly_ball_rate",
    "popup_rate",
    "risp_rate",
    "high_leverage_rate",
    "ahead_count_rate",
    "behind_count_rate",
    "avg_velocity_seen",
    "pct_vs_lefties",
    "batter_side",
]

batter_overall_df = batter_overall_df.reset_index()

# add batter names
batter_overall_df = batter_overall_df.merge(batter_name_lookup, on="batter", how="left")

# merge in count-specific features
batter_overall_df = batter_overall_df.merge(count_features, on="batter", how="left")

# merge in handedness features
batter_overall_df = batter_overall_df.merge(handedness_pivot, on="batter", how="left")

# add usage-weighted pitch type performance
pitch_type_weights = (
    batter_by_pitch_type_df.groupby("batter")
    .agg(
        {
            "pitches_seen": "sum",
            "whiff_rate": lambda x: np.average(
                x.values,
                weights=batter_by_pitch_type_df.loc[
                    x.index, "pitch_type_seen_pct"
                ].values,
            ),
            "contact_rate": lambda x: np.average(
                x.values,
                weights=batter_by_pitch_type_df.loc[
                    x.index, "pitch_type_seen_pct"
                ].values,
            ),
            "avg_exit_velocity": lambda x: np.average(
                x.values,
                weights=batter_by_pitch_type_df.loc[
                    x.index, "pitch_type_seen_pct"
                ].values,
            ),
            "barrel_rate": lambda x: np.average(
                x.values,
                weights=batter_by_pitch_type_df.loc[
                    x.index, "pitch_type_seen_pct"
                ].values,
            ),
        }
    )
    .round(3)
)

pitch_type_weights.columns = [
    "total_pritches_check",
    "weighted_whiff_rate",
    "weighted_contact_rate",
    "weighted_avg_exit_velocity",
    "weighted_barrel_rate",
]
pitch_type_weights = pitch_type_weights.reset_index()

batter_overall_df = batter_overall_df.merge(
    pitch_type_weights[
        [
            "batter",
            "weighted_whiff_rate",
            "weighted_contact_rate",
            "weighted_avg_exit_velocity",
            "weighted_barrel_rate",
        ]
    ],
    on="batter",
    how="left",
)

# add derived quality scores
batter_overall_df["contact_quality_score"] = (
    batter_overall_df["hard_contact_rate"] * 0.4
    + batter_overall_df["barrel_rate"] * 0.4
    + batter_overall_df["avg_exit_velocity"] / 100 * 0.2
).round(3)

batter_overall_df["disipline_score"] = (
    (1 - batter_overall_df["chase_rate"]) * 0.6
    + batter_overall_df["zone_swing_rate"] * 0.4
).round(3)

batter_overall_df["power_score"] = (
    batter_overall_df["power_contact_rate"] * 0.5
    + batter_overall_df["fly_ball_rate"] * 0.3
    + (batter_overall_df["avg_launch_angle"] / 30) * 0.2
).round(3)

print(f"\nOverall Batter Summary DataFrame shape: {batter_overall_df.shape}")

# save overall batter summary
batter_overall_df.to_parquet(
    "data/processed/batter_overall_summary.parquet", index=False
)
print("Overall Batter Summary saved to data/processed/batter_overall_summary.parquet")

##############################################
# Summary
##############################################
print(f"\n{'=' * 60}")
print("BATTER DATA STRUCTURE SUMMARY")
print(f"{'=' * 60}")
print(
    f"1. Batter by Pitch Type: {batter_by_pitch_type_df.shape[0]:,} rows, {batter_by_pitch_type_df.shape[1]} columns"
)
print("  -How each batter performs agains specific pitch types")
print("  -File: batter_by_pitch_type_profiles.parquet")
print(
    f"2. Batter vs Pitcher Handedness: {handedness_pivot.shape[0]:,} rows, {handedness_pivot.shape[1]} columns"
)
print("  -Performance splits vs LHP/RHP")
print("  -File: batter_vs_pitcher_handedness.parquet")
print(
    f"3. Overall Batter Summary: {batter_overall_df.shape[0]:,} rows, {batter_overall_df.shape[1]} columns"
)
print("  -Comprehensive batter profiles with quality scores")
print("  -File: batter_overall_summary.parquet")
print(f"{'=' * 60}\n")

# display sample of key batter metrics
print("\nSample Batter Overall Stats:")
key_cols = [
    "batter",
    "formatted_name",
    "overall_woba",
    "avg_exit_velocity",
    "barrel_rate",
    "chase_rate",
    "contact_quality_score",
    "disipline_score",
    "power_score",
]
print(batter_overall_df[key_cols].head(10))
