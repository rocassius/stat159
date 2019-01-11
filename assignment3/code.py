import math
import os
import warnings

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm_notebook as tqdm


####################################################################################
# UTILITY FUNCTIONS
####################################################################################

def download_and_cache_file(name, url, force_download=False):
    """
    Download and cache a file of given NAME from the given URL. The file
    will be cached in the data folder.
    A progress bar will be displayed if run within Jupyter notebook.

    :param name:            Name of the file
    :param url:             Download url of the file
    :param force_download:  Whether ignore local cache and re-download the file
    :return:                Path to the local file
    """
    cache_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', name)
    if not os.path.isfile(cache_file) or force_download:
        print("Downloading {} ...".format(os.path.basename(cache_file)))
        r = requests.get(url, stream=True)
        total_size, block_size = int(r.headers.get('content-length', 0)), 1024
        wrote = 0
        with open(cache_file, 'wb') as f:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=ImportWarning)
                warnings.simplefilter(action='ignore', category=DeprecationWarning)
                for data in tqdm(r.iter_content(block_size),
                                 total=math.ceil(total_size // block_size), unit_scale=True):
                    wrote = wrote + len(data)
                    f.write(data)
        print("File {} downloaded from {}".format(os.path.basename(cache_file), url))
    return cache_file


####################################################################################
# FUNCTIONS FOR PROCESSING WEATHER DATA
####################################################################################

def load_weather_data(stations, url="https://osf.io/qvyw5/download", start_year=None, end_year=None):
    """
    Load all weather data for the give STATIONS from the given FILE

    :param stations:    A list of stations
    :param url:         Download URL of the csv file
    :param start_year:  The start year of the desired time range
    :param end_year:    The start year of the desired time range
    :return:            Weather data represented as a list of records.
                        Each record is represented as a dictionary with keys:
                        {ID, YEARMONTHDAY, ELEMENT, DATA VALUE, M-FLAG,
                        Q-FLAG, S-FLAG, OBS-TIME}
    """
    # Download file if not exist
    cache_file = download_and_cache_file("weather_data_ca.csv", url)

    # Load data from csv file
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        weather_data = pd.read_csv(cache_file, index_col=0, low_memory=False)

    # Filter weather data based on given list of stations
    stations = pd.DataFrame(stations)
    weather_data = pd.merge(stations[['ID']], weather_data, on='ID')

    # Filter weather data based on time period
    date = pd.to_datetime(weather_data['YEARMONTHDAY'], format='%Y%m%d')
    if start_year:
        weather_data = weather_data[date >= pd.Timestamp(year=start_year, month=1, day=1)]
    if end_year:
        weather_data = weather_data[date < pd.Timestamp(year=end_year + 1, month=1, day=1)]

    return weather_data.to_dict(orient='records')


def get_temperature_data(weather_data, element_type='TMAX'):
    """
    Get max temperature data from the list of weather data records
    Convert the data value to correct unit (Fahrenheit)

    :param weather_data:  List of weather data records
    :param element_type:  Element type for max temperature records
    :return:              Max temperature data represented as a list of records.
                          Each record is represented as a dictionary with keys:
                          {ID, YEARMONTHDAY, DATA VALUE}
    """
    # Load data
    weather_data = pd.DataFrame(weather_data,
                                columns=["ID", "YEARMONTHDAY", "ELEMENT", "DATA VALUE",
                                         "M-FLAG", "Q-FLAG", "S-FLAG", "OBS-TIME"])
    result = weather_data.loc[weather_data['ELEMENT'] == element_type, ['ID', 'YEARMONTHDAY', 'DATA VALUE']]
    # Convert unit to Fahrenheit
    result['DATA VALUE'] = (result['DATA VALUE'] / 10 * 9 / 5) + 32
    return result.to_dict(orient='records')


def get_precipitation_data(weather_data, element_type='TMAX'):
    """
    Get precipitation data from the list of weather data records
    Convert the data value to correct unit (mm)

    :param weather_data:  List of weather data records
    :param element_type:  Element type for precipitation records
    :return:              Precipitation data represented as a list of records.
                          Each record is represented as a dictionary with keys:
                          {ID, YEARMONTHDAY, DATA VALUE}
    """
    # Load data
    weather_data = pd.DataFrame(weather_data,
                                columns=["ID", "YEARMONTHDAY", "ELEMENT", "DATA VALUE",
                                         "M-FLAG", "Q-FLAG", "S-FLAG", "OBS-TIME"])
    result = weather_data.loc[weather_data['ELEMENT'] == element_type, ['ID', 'YEARMONTHDAY', 'DATA VALUE']]
    # Convert unit to mm
    result['DATA VALUE'] = result['DATA VALUE'] / 10
    return result.to_dict(orient='records')


def filter_weather_data(weather_data):
    """
    Filters the weather data by requiring that for any given weather report, the maximum temperature is greater than
    the minimum temperature

    :param weather_data:(List of dictionaries) Dictionary keys contains weather station id, temperature, etc

    :return: weather_data(List of dictionaries) Dictionary keys contains weather station id, temperature, etc
    """
    df = (pd.DataFrame(weather_data)
          .reset_index()
          .pivot_table(index=['ID', 'YEARMONTHDAY'], columns=['ELEMENT'], values=['DATA VALUE']))

    invalid = (df[(pd.notna(df[('DATA VALUE', 'TMIN')])) &
                  (pd.notna(df[('DATA VALUE', 'TMAX')])) &
                  (df[('DATA VALUE', 'TMAX')] < df[('DATA VALUE', 'TMIN')])].reset_index()[['ID', 'YEARMONTHDAY']])
    invalid.columns = invalid.columns.droplevel(1)
    invalid_index = set([tuple(value + [element])
                         for value in invalid.values.tolist()
                         for element in ['TMAX', 'TMIN']])

    return list(filter(lambda record: (record['ID'], record['YEARMONTHDAY'], record['ELEMENT']) not in invalid_index,
                       weather_data))


def find_valid_stations_id_each_year(weather_data):
    """
    For each year, find the id of stations that have at least one weather record

    :param weather_data:    A list of weather data records
    :return:                A dictionary which maps from year to a set of station id
    """
    weather_data = pd.DataFrame(weather_data)

    date = pd.to_datetime(weather_data['YEARMONTHDAY'], format='%Y%m%d')
    return (pd.DataFrame({'ID': weather_data['ID'], 'YEAR': date.dt.year})
            .groupby('YEAR')['ID']
            .agg(lambda ids: set(ids))
            .to_dict())


def compute_station_bias(weather_data, iters=500):
    """
    Compute the bias adjustment for each weather station

    :param weather_data:    A list of max temperature or precipitation records
    :param iters:           Number of iterations to run
    :return:                A dictionary mapping from station id to its bias value
    """
    weather_data = (pd.DataFrame(weather_data, columns=['ID', 'YEARMONTHDAY', 'DATA VALUE'])
                    .pivot_table(values='DATA VALUE', index='ID', columns='YEARMONTHDAY'))

    bias = np.zeros(weather_data.shape[0])
    for _ in range(iters):
        i = np.random.randint(len(bias))
        bias_update = np.zeros(len(bias))
        for j in range(len(bias)):
            bias_update[j] = ((weather_data.iloc[i] + bias[i]) - (weather_data.iloc[j] + bias[j])).mean()
        bias += np.nan_to_num(bias_update)
        bias -= np.mean(bias)
    return dict(zip(weather_data.index, bias))


def apply_station_bias(weather_data, bias):
    """
    Apply the station bias to weather data

    :param weather_data:    A list of max temperature or precipitation records
    :param bias:            A dictionary mapping from station id to its bias value
    :return:                A list of bias-adjusted weather data dicts
    """
    result = []
    for data in weather_data:
        data['DATA VALUE'] += bias[data['ID']]
        result.append(data)
    return result


def get_station_weights(stations, distances):
    """
    Construct a dictionary with matching station id and its corresponding weight

    :param stations:        A list of station records
    :param distances:       A list of weights
    :return:                A dictionary with matching station id and its corresponding weight
    """
    station_weights = {}
    stations_ids = [dic['ID'] for dic in stations]
    for i in range(0, len(distances)):
        station_weights[stations_ids[i]] = distances[i]
    return station_weights


def aggregate_temperature_data(temperature_data, station_weights):
    """
    Construct 11 bins (different temperature range)
    with each bin having a count on each month of each year

    :param temperature_data:    A list of max temperature data dicts
    :param station_weights:     A dictionary with matching station id and its corresponding weight
    :return:                    A list of lists that contain counts for each bin
    """

    temperature_table = pd.DataFrame(temperature_data)
    weights = []
    for index, row in temperature_table.iterrows():
        weights.append(station_weights[row['ID']])
    temperature_table['Station_Inverse_Weights'] = weights
    wtavg = lambda x: np.average(x.iloc[:, 0], weights=x.Station_Inverse_Weights, axis=0)
    temperature_bin_table = temperature_table.groupby(['YEARMONTHDAY']).apply(wtavg).to_frame().reset_index().rename(
        columns={0: 'Weighted_Average_Temperature'})
    temperature_bin_table['Year_Month'] = pd.to_datetime(temperature_bin_table.YEARMONTHDAY,
                                                         format='%Y%m%d').dt.to_period("M")
    cats = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-110']
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    df2 = (temperature_bin_table.groupby(
        ['Year_Month', pd.cut(temperature_bin_table['Weighted_Average_Temperature'], bins, labels=cats)])
           .size()
           .unstack(fill_value=0)
           .reindex(columns=cats, fill_value=0))
    bins_count = []
    for column in df2:
        bins_count.append(df2[column].values.tolist())
    return bins_count


def aggregate_precipitation_data(precipitation_data, station_weights):
    """
    Construct 5 bins (different precipitation level)
    with each bin having a count on each month of each year

    :param precipitation_data:  A list of max precipitation data dicts
    :param station_weights:     A dictionary with matching station id and its corresponding weight
    :return:                    A list of lists that contain counts for each bin
    """

    precipitation_table = pd.DataFrame(precipitation_data)
    weights = []
    for index, row in precipitation_table.iterrows():
        weights.append(station_weights[row['ID']])
    precipitation_table['Station_Inverse_Weights'] = weights
    wtavg = lambda x: np.average(x.iloc[:, 0], weights=x.Station_Inverse_Weights, axis=0)
    precipitation_bin_table = precipitation_table.groupby(['YEARMONTHDAY']).apply(
        wtavg).to_frame().reset_index().rename(columns={0: 'Weighted_Average_Precipitation'})
    precipitation_bin_table['Year_Month'] = pd.to_datetime(precipitation_bin_table.YEARMONTHDAY,
                                                           format='%Y%m%d').dt.to_period("M")
    cats = ['<1', '1-5', '5-15', '15-29', '>29']
    bins = [-1, 1, 5, 15, 29, 1000]
    df3 = (precipitation_bin_table.groupby(
        ['Year_Month', pd.cut(precipitation_bin_table['Weighted_Average_Precipitation'], bins, labels=cats)])
           .size()
           .unstack(fill_value=0)
           .reindex(columns=cats, fill_value=0))
    bins_count = []
    for column in df3:
        bins_count.append(df3[column].values.tolist())
    return bins_count
