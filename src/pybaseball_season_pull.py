# This script is intended to test the functionality of the pybaseball library
# pip install git+https://github.com/jldbc/pybaseball.git
import pybaseball
import pandas as pd
# from datetime import date, timedelta


def fetch_statcast_data(
    start_dt,
    end_dt,
    output_dir="/Users/jessi/Documents/GitHub Repositories/Pitcher-Hitter-Analysis/data/raw",
):
    """
    Fetch Statcast data for the given date range and saves it to Parquet files.

    Args:
        start_dt (str): Start date in 'YYYY-MM-DD' format.
        end_dt (str): End date in 'YYYY-MM-DD' format.
        output_dir (str): Directory to save the Parquet files.
    """
    print(f"Fetching Statcast data from {start_dt} to {end_dt}...")
    try:
        # pybaseball's statcast() function handles chucnking for large date ranges
        data = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt)

        if not data.empty:
            # Create output directory if it doesn't exist
            import os

            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(
                output_dir, f"statcast_raw_{start_dt}_{end_dt}.parquet"
            )
            data.to_parquet(file_path, index=False)
            print(f"Successfully saved {len(data)} rows to {file_path}")
        else:
            print(f"No data found for the given date range: {start_dt} to {end_dt}")

    except Exception as e:
        print(f"Error fetching Statcast data: {e}")


if __name__ == "__main__":
    # Define a reasonable historical range for initial data pull
    # Adjust these dates based on how much historical data you want to analyze
    start_date_historical = "2024-04-01"  # Example start date
    end_date_historical = "2024-10-30"  # Example end date

    # Fetch Statcast data for the defined date range
    fetch_statcast_data(start_date_historical, end_date_historical)

    # Grabbing multiple seasons to add to the dataset
    # not looping through years for now since this is to grab a larger set of data for initial model
    # can change to loop if needed
    # seasons = [2020, 2021, 2022, 2023]
    start_date_historical = "2023-03-30"
    end_date_historical = "2023-11-01"
    fetch_statcast_data(start_date_historical, end_date_historical)

    start_date_historical = "2022-04-07"
    end_date_historical = "2022-11-05"
    fetch_statcast_data(start_date_historical, end_date_historical)

    start_date_historical = "2021-04-01"
    end_date_historical = "2021-11-02"
    fetch_statcast_data(start_date_historical, end_date_historical)

    start_date_historical = "2020-07-23"
    end_date_historical = "2020-10-27"
    fetch_statcast_data(start_date_historical, end_date_historical)
