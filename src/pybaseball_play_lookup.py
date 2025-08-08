# This is a script intended to test the functionality of the pybaseball library and player lookup for Statcast data.
import pybaseball
import pandas as pd

# enable caching for pybaseball
pybaseball.cache.enable()

player_id = [434378]

pybaseball.playerid_reverse_lookup(player_id, key_type="mlbam")

# texas_rangers_2024_df = pd.DataFrame(pybaseball.statcast('2024-04-01', '2024-10-30', team='TEX'))

# show the first few rows of the DataFrame
# print(texas_rangers_2024_df.head())
