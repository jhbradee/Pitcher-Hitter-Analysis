import pandas as pd
import pybaseball

# setting up dataframe and other variables
raw_data_path = "/Users/jessi/Documents/GitHub Repositories/Pitcher-Hitter-Analysis/data/raw/statcast_raw_2025-07-29_2025-07-30.parquet"
july29_30_df = pd.read_parquet(raw_data_path)

# printing the first few rows of the dataframe and a summary of the columns and data types
# print("Data fetched from Parquet file:")
# print(july29_30_df.head())
# print("\nDataFrame Info:")
# print(july29_30_df.info())

# Note: The above code assumes that the Parquet file exists at the specified path.

# define a list of "at-bat ending" events we care about
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

# filter the dataframe to only include rows where the event is in our list
filtered_df = july29_30_df[july29_30_df["events"].isin(at_bat_ending_events)].copy()

# remove any rows where 'pitcher' or 'batter' is missing
filtered_df.dropna(subset=["pitcher", "batter"], inplace=True)

# creating a pitcher profile dataframe
pitcher_profile_df = (
    filtered_df.groupby(["pitcher", "player_name", "pitch_type", "pitch_name"])
    .agg(
        total_pitches=("pitch_type", "count"),
        avg_release_speed=("release_speed", "mean"),
        avg_spin_rate=("release_spin_rate", "mean"),
    )
    .reset_index()
)

total_pitch_by_type = pitcher_profile_df.groupby("pitcher")["total_pitches"].transform(
    "sum"
)
pitcher_profile_df["pitch_type_percentage"] = (
    pitcher_profile_df["total_pitches"] / total_pitch_by_type * 100
)

# print("\nPitcher Profile DataFrame:")
# print(pitcher_profile_df.head())

# creating a batter profile dataframe

unique_batters = filtered_df["batter"].unique().tolist()
batter_names = pybaseball.playerid_reverse_lookup(unique_batters, key_type="mlbam")

# Create a new column with formatted names: "Last, First" (capitalized)
batter_names["formatted_name"] = (
    batter_names["name_last"].str.capitalize()
    + ", "
    + batter_names["name_first"].str.capitalize()
)

# Select the formatted name and key_mlbam columns
formatted_batter_name = batter_names[["formatted_name", "key_mlbam"]]

batter_profile_df = (
    filtered_df.groupby(["batter", "pitch_type", "pitch_name"])
    .agg(
        avg_launch_angle=("launch_angle", "mean"),
        avg_launch_speed=("launch_speed", "mean"),
        total_at_bats=("pitch_type", "count"),
    )
    .reset_index()
)

# merge batter_profile_df with formatted_batter_name to add the formatted name
batter_profile_df = batter_profile_df.merge(
    formatted_batter_name, left_on="batter", right_on="key_mlbam", how="left"
)

batter_profile_df = batter_profile_df[
    [
        "batter",
        "formatted_name",
        "pitch_type",
        "pitch_name",
        "avg_launch_angle",
        "avg_launch_speed",
        "total_at_bats",
    ]
]

# print("\nBatter Profile DataFrame:")
# print(batter_profile_df.head())

# merge pitcher_profile_df with batter_profile_df to create a combined profile
combined_profile_df = pd.merge(
    filtered_df, pitcher_profile_df, on=["pitcher", "pitch_type"], how="left"
)
combined_profile_df = pd.merge(
    combined_profile_df, batter_profile_df, on=["batter", "pitch_type"], how="left"
)

print("\nCombined Profile DataFrame:")
print(combined_profile_df.head())

combined_profile_df.to_parquet(
    "/Users/jessi/Documents/GitHub Repositories/Pitcher-Hitter-Analysis/data/processed/combined_profile_2025-07-29_2025-07-30.parquet",
    index=False,
)
print(
    "\nCombined profile data saved to 'data/processed/combined_profile_2025-07-29_2025-07-30.parquet', Successfully cleaned, transformed, and saved the data."
)
