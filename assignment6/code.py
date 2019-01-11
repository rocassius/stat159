import math
import os
import warnings
import zipfile
from functools import partial
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
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


def parallel_list_map(lst, func, num_partitions=100, **kwargs):
    """
    Multi-threading helper to apply a function on a list. This function
    will return the same result as calling func(lst, **kwargs) directly.

    The function must take a list as input (may have other arguments) and
    return a list as its output.

    :param lst             The input list
    :param func            The function to be applied on the list
    :param num_partitions  Number of threads
    :return:               The same output list as returned by func(lst)
    """
    # Split the list based on number of partitions
    lst_split = [lst[i::num_partitions] for i in range(num_partitions)]
    # Create a thread pool
    pool = Pool(cpu_count())
    # Run the function and concatenate the result
    lst = sum(pool.map(partial(func, **kwargs), lst_split), [])
    # Clean up
    pool.close()
    pool.join()
    return lst


####################################################################################
# FUNCTIONS FOR LOADING CRIME DATA
####################################################################################

def load_crime_data(start_year=1980, end_year=2009, state_code='06', county_code='001', parallel=True):
    """
    Download the URC data file and load all files into a single panda DataFrame

    :param start_year:   The start year of the desired time range (default: 1980)
    :param end_year:     The end year of the desired time range (default: 2009)
    :param state_code:   FIPS state code (default: 06/California)
    :param county_code:  FIPS county code (default: 001/Alameda)
    :param parallel:     Load crime data using multi-threading
    :return:             A pandas DataFrame
    """
    ucr_data_file = download_and_cache_file('ucr_offenses_known_monthly_1960_2016_dta.zip',
                                            'https://www.openicpsr.org/openicpsr/project/100707/version/V7/' +
                                            'download/file?filePath=/openicpsr/100707/fcr:versions/V7' +
                                            '/ucr_offenses_known_monthly_1960_2016_dta.zip')
    year_range = list(range(start_year, end_year + 1))

    if not parallel:
        frames = load_crime_data_executor(year_range, ucr_data_file, state_code, county_code)
    else:
        frames = parallel_list_map(year_range, load_crime_data_executor,
                                   num_partitions=len(year_range), ucr_data_file=ucr_data_file,
                                   state_code=state_code, county_code=county_code)
    return pd.concat(frames).reset_index(drop=True)


def load_crime_data_executor(years, ucr_data_file, state_code, county_code):
    """
    A helper function for loading crime data from files of each year

    :param years:          A list of years which we need to load corresponding data
    :param ucr_data_file:  The UCR data zip file
    :param state_code:     FIPS state code
    :param county_code:    FIPS county code
    :return:               A list of pandas DataFrames, each corresponds to a single year
    """
    with zipfile.ZipFile(ucr_data_file, 'r') as ucr_data:
        return [clean_crime_data(pd.read_stata(
            ucr_data.open('ucr_offenses_known_monthly_{}.dta'.format(year))),
            state_code=state_code, county_code=county_code)
            for year in years]


def clean_crime_data(crime_data, state_code, county_code):
    """
    Clean up data from UCR and extract crime statistics according to Ranson's paper.

    :param crime_data:   The original data frame from UCR
    :param state_code:   FIPS state code
    :param county_code:  FIPS county code
    :return:             Cleaned data frame
    """
    # Rename crime statistics columns
    crime_data.rename({
        'act_murder': 'murder',
        'act_manslaughter': 'manslaughter',
        'act_rape_total': 'rape',
        'act_aggravated_assault': 'aggravated_assault',
        'act_simple_assault': 'simple_assault',
        'act_robbery_total': 'robbery',
        'act_burglary_total': 'burglary',
        'act_theft_total': 'larceny',
        'act_mtr_vhc_theft_total': 'vehicle_theft'
    }, axis='columns', inplace=True)

    # Convert number columns to int (set to 0 for missing values)
    for column in ['zip_code', 'murder', 'manslaughter', 'rape', 'aggravated_assault',
                   'simple_assault', 'robbery', 'burglary', 'larceny', 'vehicle_theft']:
        crime_data[column] = (pd.to_numeric(crime_data[column], errors='coerce')
                              .fillna(0).astype('int32'))

    # Convert month to int
    crime_data['month'] = pd.to_datetime(crime_data['month'], format='%B').dt.month

    # Filter state and county
    crime_data = crime_data[crime_data['fips_state_county_code'] == state_code + county_code]

    return crime_data[['ori', 'year', 'month', 'zip_code', 'murder', 'manslaughter', 'rape',
                       'aggravated_assault', 'simple_assault', 'robbery', 'burglary', 'larceny',
                       'vehicle_theft']]


def generate_crime_dict(crime_data_df):
    """
    Get the vector of counts of each month for all crimes

    :param crime_data_df:  The cleaned crime_data dataframe
    :return:               Mapping from crime name to the counter vector
    """
    crime_name = ['murder', 'manslaughter', 'rape', 'aggravated_assault',
                  'simple_assault', 'robbery', 'burglary', 'larceny',
                  'vehicle_theft']
    grouped_crime_data = crime_data_df.groupby(["year", "month"]).sum().reset_index()
    return {name: grouped_crime_data[name].values for name in crime_name}


def add_lag(lst):
    """
    Adding the values of previous index to the current index with the exception of first index adding itself.

    :param lst: a list
    :return: a list with values added to it
    """
    lag = lst[:-1]
    lag.insert(0, lst[0])
    return (np.asarray(lag) + np.asarray(lst)).tolist()


def generate_combined_df(crime_dict, bins_T, bins_P):
    """
    Generate a list of lists with the first part being the total crime count for 360 months, 
    the second part being the different categories of temperature data, and the last part 
    being the different categories of precipitation data.

    :param crime_dict: a dictionary
    :param bins_T: a list of lists
    :param bins_P: a list of lists

    :return: a list of lists
    """
    crime_names = ['murder', 'manslaughter', 'rape', 'aggravated_assault',
                   'simple_assault', 'robbery', 'burglary', 'larceny', 'vehicle_theft']
    dict_len = len(crime_dict['murder'])
    total_crime_count = np.asarray([0] * dict_len)
    for crime in crime_names:
        total_crime_count += crime_dict[crime]
    total_crime_count = total_crime_count.tolist()
    T = [add_lag(b) for b in bins_T]
    P = [add_lag(b) for b in bins_P]
    return [total_crime_count] + T + P


def generate_one_hot_encoding_df(df):
    """
    Generate dummy values of 1 or 0 for each row. For example, a row in column 1 would be 1 
    if that row corresponds to January of some years and 0 otherwise.

    :param df: a panda dataframe with columns such as total crime count by month_year and weather data
    :return: a concatenated panda dataframe with new dummy columns added to the original data frame.
    """
    months = pd.get_dummies(pd.to_numeric(df.index.str[5:]))
    years = pd.get_dummies(pd.to_numeric(df.index.str[:4]))
    months.index = df.index
    years.index = df.index

    return pd.concat([df, months, years], axis=1)


def split_crime_data(crime_df):
    """
    Split crime data of Alameda County to the east and the west by zipcodes

    :param crime_df: a panda dataframe with columns being the count of different crimes by zipcode
    
    :return: two panda dataframes with one being the crime dataframe for the west of Alameda County,
    and the other one being the crime dataframe for the east of Alameda County
    """
    east_zipcodes = [94568, 94588, 94566, 94550, 94586, 94551, 94514]
    east_crime = crime_df[crime_df['zip_code'].isin(east_zipcodes)]
    west_crime = crime_df[~crime_df['zip_code'].isin(east_zipcodes)]
    return east_crime, west_crime


def get_df_permutation(df1, df2):
    """
    Do permutation by swapping the rows between two dataframe with 0.5
    probability

    :param df1: a panda dataframe
    :param df2: a panda dataframe
    
    :return:    input panda dataframes after permutation
    """
    df1, df2 = df1.copy(), df2.copy()
    assert df1.shape == df2.shape
    for row in range(df1.shape[0]):
        if np.random.random() < 0.5:
            df2.iloc[row, :], df1.iloc[row, :] = \
                df1.iloc[row, :].copy(), df2.iloc[row, :].copy()
    return df1, df2


def get_rms(model, df, y):
    """
    Get the RMSE for a stats.models model and new data

    :param model:  a stats.models model
    :param df:     pandas dataframe containing all the data
    :param y:      (array-like) the true responses of the response variable
    
    :return:       a numeric RMSE
    """
    result = model.fit()
    predictions = result.predict(df)
    return rmse(predictions, y)


def get_permutation_stats(df1, df2, y, reps):
    """
    Get the test statistics of permutations 

    :param df1:    (pandas dataframe) contains crime weather data for one region
    :param df2:    (pandas dataframe) contains crime weather data for one region
    :param y:      (string) the name of the response variable
    :param reps:   (int) the number of permutations to make
    
    :return:       a list of the computed statistics
    """

    y1, y2 = df1[y], df2[y]
    df1_x = df1[df1.columns.difference([y])]
    df2_x = df2[df2.columns.difference([y])]

    stats = [0 for _ in range(reps)]
    for i in range(reps):
        # Make the permutation in crime-weather data
        df1_xp, df2_xp = get_df_permutation(df1_x, df2_x)

        # Add the one hot encodings
        poisson_df1 = generate_one_hot_encoding_df(df1_xp)
        poisson_df2 = generate_one_hot_encoding_df(df2_xp)

        # Make both Poisson Regression Models
        model_1 = sm.GLM(y1, poisson_df1, family=sm.families.Poisson())
        model_2 = sm.GLM(y2, poisson_df1, family=sm.families.Poisson())

        # Make Predictions on the training data and get RMSEs for the training data
        rms_1 = get_rms(model_1, poisson_df1, y1)
        rms_2 = get_rms(model_2, poisson_df2, y1)

        # Combine the two RMSEs into one RMSE
        stat = np.sqrt(0.5*(rms_1 ** 2 + rms_2 ** 2))

        stats[i] = stat

    return stats
