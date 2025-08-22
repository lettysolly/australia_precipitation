# Functions to help with reading and working with NetCDF data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime
import xarray as xr


# function for reading station metadata into a pandas df:
def read_stations_metadata(nc_files):
    '''
    Function builds a dataframe containing station metadata from a list of netcdf files
    parameter: nc_files - a list containing the file paths within the NetCDF directory
    '''
    station_info = []

    for file_path in nc_files:
        ds = nc.Dataset(file_path)

        # Read all station metadata (assuming netcdf files contain this info)
        station_id = getattr(ds, 'Station_Number')
        station_name = getattr(ds, 'Station_Name').strip()
        state = getattr(ds, 'State').strip()
        lat = getattr(ds, 'Latitude')
        lon = getattr(ds, 'Longitude')
        first_year = getattr(ds, 'First_year')
        latest_year = getattr(ds, 'Lastest_year')
        completeness = getattr(ds, 'Percentage_of_complete')

        # Add all station info to the df
        station_info.append({
            'Station Number': station_id,
            'Station Name': station_name,
            'State': state,
            'Latitude': lat,
            'Longitude': lon,
            'Start Year': first_year,
            'End Year': latest_year,
            'Percentage Complete (from file)': completeness
            
        })
                
        ds.close()

    # Make the dataframe
    station_data_df = pd.DataFrame(station_info)
    
    # Numericalise the numerical variables
    station_data_df['Percentage Complete (from file)'] = pd.to_numeric(station_data_df['Percentage Complete (from file)'], errors='coerce')
    station_data_df['Percentage Complete (from file)'] = station_data_df['Percentage Complete (from file)'].fillna(0)
    station_data_df['Start Year'] = pd.to_numeric(station_data_df['Start Year'], errors='coerce')
    station_data_df['End Year'] = pd.to_numeric(station_data_df['End Year'], errors='coerce')

    # Compute Record Length (in years)
    station_data_df['Record Length'] = station_data_df['End Year'] - station_data_df['Start Year'] + 1

    return station_data_df

# function for building dataframes containing rain timeseries for specific lists of states:
def build_rain_dataframe(stations, nc_dir, state_abbrev, min_rain=0.01):
    """
    Build a rainfall DataFrame from NetCDF files for a list of stations.
    
    Parameters
    ----------
    stations : list
        List of station numbers.
    nc_dir : str
        Path to directory containing NetCDF files.
    state_abbrev : str
        State abbreviation in full caps (e.g., "VIC", "NSW").
    min_rain : float
        Minimum rainfall to keep (values below this become NaN).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index, station numbers as columns,
        rainfall values (NaN for < min_rain), only rows with at least one value,
        sorted in chronological order.
    """
    
    all_data = []
    
    for station_num in stations:
        # Assuming files are named like STATE_stationnumber.nc
        file_path = os.path.join(nc_dir, f"{state_abbrev}_{station_num}.nc")
        
        if not os.path.exists(file_path):
            print(f"No file found for station {station_num}")
            continue
        
        # Open NetCDF
        ds = nc.Dataset(file_path)
        
        # Time variable
        time_var = ds.variables["time"]
        times = nc.num2date(time_var[:], units=time_var.units)
        times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in times]
        
        # Rain variable
        rain = ds.variables['prcp_inst'][:]
        
        # Filter out small rainfall values
        rain = pd.Series(rain)
        rain = rain.where(rain >= min_rain, other=float('nan'))
        
        # Build DataFrame for this station
        df_station = pd.DataFrame({station_num: rain.values}, index=pd.to_datetime(times))
        df_station = df_station[~df_station.index.duplicated(keep='first')]
        
        all_data.append(df_station)
        ds.close()
    
    # Combine all stations, drop all-NaN rows, sort by datetime
    df_all = pd.concat(all_data, axis=1)
    df_all = df_all.dropna(how="all")
    df_all = df_all.sort_index()
    
    return df_all


# Function for summing rainfall either on daily or yearly timescale

def rainfall_sum(nc_files, timescale='yearly'):
    """
    Calculate rainfall sums from half-hourly station NetCDF files.

    Arguments: 
    - nc_files: list of NetCDF file paths
    - timescale: 'yearly' or 'daily'
    """
    data = {}

    for file in nc_files:
        ds = xr.open_dataset(file)
        rain = ds["prcp_inst"]

        # Convert to pandas with datetime index
        df = rain.to_pandas()
        df.index = pd.to_datetime(df.index)

        # Resample to daily totals first
        daily = df.resample("D").sum(min_count=1)

        if timescale == 'daily':
            out = daily
        elif timescale == 'yearly':
            yearly = daily.resample("YE").sum(min_count=1)
            yearly.index = yearly.index.year
            out = pd.Series(yearly.squeeze(), index=yearly.index)
        else:
            raise ValueError("timescale must be 'daily' or 'yearly'")

        # Station ID
        station_id = ds.attrs.get("Station_Number")
        data[station_id] = out

        ds.close()

    # Combine into one DataFrame
    combined_df = pd.concat(data, axis=1)
    combined_df = combined_df.sort_index()
    return combined_df



