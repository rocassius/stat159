from functools import partial
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from Crypto.Random.random import getrandbits

TEMPERATURE_BINS_LABEL = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60',
                          '60-70', '70-80', '80-90', '90-100', '100-110']
TEMPERATURE_BINS_CUTOFF = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]


####################################################################################
# UTILITY FUNCTIONS
####################################################################################

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


# ===================#
# =*= Functions =*= #
# ===================#

def bin_temperature_data(temperature_data):
    """
    Retrieve and bin maximum temperature data for all stations

    :param temperature_data: (list) Max temperature data represented as a list of records.
                             Each record is represented as a dictionary with keys:
                             {ID, YEARMONTHDAY, DATA VALUE}

    :return:                 (dictionary) A mapping from station ID to each stations
                             binned temperature data.
                             The temperature data is represented as a dictionary mapping from
                             date (YEARMONTHDAY) to temperature bin labels.
    """
    return (pd.DataFrame(temperature_data)
            .pivot_table(columns='ID', index='YEARMONTHDAY', values='DATA VALUE')
            .apply(lambda x: pd.cut(x, TEMPERATURE_BINS_CUTOFF, labels=TEMPERATURE_BINS_LABEL))
            .to_dict())


def remove_na_from_data(temp_daily1, temp_daily2):
    """
    Remove all dates which at least one of the stations reports NA

    :param temp_daily1:   (dictionary) The temperature data for one station
                           mapping from date to temperature bins
    :param temp_daily2:   (dictionary) The temperature data for one station
                           mapping from date to temperature bins
    :return:              (list of dictionary) Filtered list of daily temperature data for stations
    """
    df = pd.DataFrame({i: pd.Series(data) for i, data in
                       enumerate([temp_daily1, temp_daily2])})
    df = df[df.notna().all(axis=1)]
    return [df[i].to_dict() for i in range(df.shape[1])]


def get_weather_permutation(temp_daily1, temp_daily2):
    """
    Randomly switch weather data for a selected subset of dates
    :param temp_daily1:   (dictionary) The temperature data for one station
                           mapping from date to temperature bins
    :param temp_daily2:   (dictionary) The temperature data for one station
                           mapping from date to temperature bins
    :return:              A permutation of the original two daily temperature dictionaries
    """
    assert temp_daily1.keys() == temp_daily2.keys()

    random_bits = getrandbits(len(temp_daily1.keys()))
    for key in temp_daily1.keys():
        if random_bits & 1:
            temp_daily1[key], temp_daily2[key] = temp_daily2[key], temp_daily1[key]
        random_bits = random_bits >> 1
    return temp_daily1, temp_daily2

def aggregate_bin_temperature_by_month(temp_daily):
    """
    Aggregate the binned temperature data by month and count number of days
    in each temperature bin

    :param temp_daily:   (dictionary) The temperature data for one station
                          mapping from date to temperature bins
    :return:             (dictionary) Number of days in each temperature bin
                          in each month. {Month => {Temperature Bin => #days}}
    """
    # # Using pandas (deprecated due to inefficiency)
    # df = pd.DataFrame({'bin': pd.Series(temp_daily)})
    # df['month'] = pd.to_datetime(df.index.to_series(), format='%Y%m%d').dt.strftime('%Y-%m')
    # df['count'] = 1
    # result = df.pivot_table(columns='month', index='bin', values='count', fill_value=0, aggfunc=sum)
    # for bin_label in TEMPERATURE_BINS_LABEL:
    #     if bin_label not in result.columns:
    #         result[bin_label] = 0
    # return result[TEMPERATURE_BINS_LABEL].to_dict()

    all_months = set(map(lambda x: x // 100, temp_daily.keys()))

    counter = {month: {bin_label: 0 for bin_label in TEMPERATURE_BINS_LABEL} for month in all_months}
    for date, bin_label in temp_daily.items():
        counter[date // 100][bin_label] += 1
    return counter


def compute_p_values(temp_monthly1, temp_monthly2):
    """
    Compute the p_values according to a chi square contingency tests for each month.
    
    :param temp_monthly1: (dictionary) Mapping from bin labels to a dictionary
                           containing number of days in that bin for each month
    :param temp_monthly2: (dictionary) Mapping from bin labels to a dictionary
                           containing number of days in that bin for each month
    
    :return:               (list) list of floats, the p_values
    """

    # Get months for which they both report temperature data
    months = set(temp_monthly1.keys()) & set(temp_monthly2.keys())

    p_values = []
    for month in months:
        row1 = [temp_monthly1[month][label] for label in TEMPERATURE_BINS_LABEL]
        row2 = [temp_monthly2[month][label] for label in TEMPERATURE_BINS_LABEL]

        # make a contingency table ensuring that no cell has an expectation of zero
        table = eliminate_zero_expectations([row1, row2])

        # Get the p value from a chi square contingency test
        p_value = chi2_contingency(table)[1]
        p_values.append(p_value)

    return p_values


def eliminate_zero_expectations(obs):
    """
    Make an adjusted observed table that is free of 0 expectations

    :param obs: (matrix-like) Matrix of observed values

    :return:    (matrix-like) Matrix of observed values excluding rows of zeros
    """

    obs = np.array(obs)
    new = [row for row in obs.T if not (sum(row) == 0)]
    new = np.asarray(new).T
    return new


def fisher_combine(probs):
    """
    Return the result of Fisher's combining function

    :param probs: (array-like) List of weather data records
  
    :return:      (float) The output of Fisher's combining function
    """
    return -2 * sum(np.log(np.asarray(probs)))


def permutation_test(temp_daily1, temp_daily2, reps=200):
    """
    Perform a permutation test on two sets of weather data

    :param temp_daily1:   (dictionary) The temperature data for one station
                           mapping from date to temperature bins
    :param temp_daily2:   (dictionary) The temperature data for one station
                           mapping from date to temperature bins
    :param reps:          (int)  number of permutations to make

    :return:              (list) list of numerical statistics
    """
    # Allocate storage for the final statistics
    stats = [0 for _ in range(reps)]
    return permutation_test_executor_parallel(stats, temp_daily1, temp_daily2)


def permutation_test_executor(stats, temp_daily1, temp_daily2):
    for i in range(len(stats)):
        daily_1, daily_2 = get_weather_permutation(temp_daily1.copy(), temp_daily2.copy())

        # Aggregate into monthly data
        monthly_1 = aggregate_bin_temperature_by_month(daily_1)
        monthly_2 = aggregate_bin_temperature_by_month(daily_2)

        # Compute monthly p values for this particular permutation
        p_values = compute_p_values(monthly_1, monthly_2)

        # Combine the p values into a single statistic
        stats[i] = fisher_combine(p_values)

    return stats


def permutation_test_executor_parallel(stats, temp_daily1, temp_daily2):
    return parallel_list_map(
        stats, permutation_test_executor, temp_daily1=temp_daily1, temp_daily2=temp_daily2)
