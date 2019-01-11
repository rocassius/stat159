import unittest

from code import *

CRIME = load_crime_data(start_year=1980, end_year=1981)


class TestFunctions(unittest.TestCase):
    def test_download_and_cache_file(self):
        path = download_and_cache_file("stations_ca.csv",
                                       "https://osf.io/ev7y2/download",
                                       force_download=True)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.basename(path), "stations_ca.csv")

    def test_load_crime_data(self):
        crime_data_loaded_with_single_thread = load_crime_data(
            start_year=1980, end_year=1981, parallel=False)
        self.assertTrue(CRIME.equals(crime_data_loaded_with_single_thread))

        num_rows, num_cols = CRIME.shape
        self.assertEqual(num_rows, 552)
        self.assertEqual(num_cols, 13)

        cols_name = list(CRIME)
        self.assertEqual(set(cols_name),
                         {'ori', 'year', 'month', 'zip_code', 'murder',
                          'manslaughter', 'rape', 'aggravated_assault',
                          'simple_assault', 'robbery', 'burglary',
                          'larceny', 'vehicle_theft'})
        min_year, max_year = CRIME['year'].min(), CRIME['year'].max()
        self.assertEqual(min_year, 1980)
        self.assertEqual(max_year, 1981)

    def test_generate_crime_dict(self):
        crime_dict = generate_crime_dict(CRIME)
        self.assertEqual(type(crime_dict), dict,
                         "Type of result should be dict")

        keys = list(crime_dict.keys())
        self.assertEqual(keys, ['murder', 'manslaughter', 'rape',
                                'aggravated_assault', 'simple_assault',
                                'robbery', 'burglary', 'larceny',
                                'vehicle_theft'])

        for key in keys: self.assertEqual(len(crime_dict[key]), 24)

    def test_generate_combined_df(self):
        crime_dict = generate_crime_dict(CRIME)
        leng  = len(crime_dict['murder'])
        bins_T = np.asarray([1] * leng).tolist()
        bins_T = [bins_T] * 11
        bins_P = np.asarray([2] * leng).tolist()
        bins_P = [bins_P] * 3
        combined_df = generate_combined_df(crime_dict, bins_T, bins_P)

        self.assertEqual(type(combined_df), list,
                        "Type of result should be list")
        self.assertEqual(type(combined_df[0]), list,
                        "Type of result should be list")
        self.assertEqual(len(combined_df), 15)
        self.assertEqual(len(combined_df[1]), leng)
        self.assertEqual(len(combined_df[2]), leng)

    def test_generate_one_hot_encoding_df(self):
        df = pd.DataFrame(data={'Year_Month': ["1980-1", "1981-2"]})
        df.set_index('Year_Month', inplace=True)
        df_encoded = generate_one_hot_encoding_df(df)

        num_rows, num_cols = df_encoded.shape
        self.assertEqual(num_rows, 2)
        self.assertEqual(num_cols, 4)

        cols_name = list(df_encoded)
        self.assertEqual(cols_name, [1, 2,
                                     1980, 1981])

        self.assertEqual(df_encoded.loc['1980-1', 1980], 1)
        self.assertEqual(df_encoded.loc['1980-1', 1], 1)
        self.assertEqual(df_encoded.loc['1980-1', 1981], 0)
        self.assertEqual(df_encoded.loc['1980-1', 2], 0)
        self.assertEqual(df_encoded.loc['1981-2', 1980], 0)
        self.assertEqual(df_encoded.loc['1981-2', 1], 0)
        self.assertEqual(df_encoded.loc['1981-2', 1981], 1)
        self.assertEqual(df_encoded.loc['1981-2', 2], 1)

    def test_split_crime_data(self):
        mock_crime_df = pd.DataFrame(data={'zip_code': [
            94568, 94588, 94566, 94550, 94586, 94551, 94514, 94704, 90210]})
        east_zipcodes = [94568, 94588, 94566, 94550, 94586, 94551, 94514]

        east_crime, west_crime = split_crime_data(mock_crime_df)
        self.assertEqual(east_crime.shape, (7, 1))
        self.assertEqual(west_crime.shape, (2, 1))
        self.assertTrue(all(east_crime['zip_code'].isin(east_zipcodes)))
        self.assertTrue(all(~west_crime['zip_code'].isin(east_zipcodes)))

    def test_add_lag(self):
        self.assertEqual(add_lag([1, 2, 3, 4, 5]), [2, 3, 5, 7, 9])
        self.assertEqual(add_lag([1, 1, 1, 1, 1]), [2, 2, 2, 2, 2])

    def test_get_df_permutation(self):
        df1 = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        df2 = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        perm_df1, perm_df2 = get_df_permutation(df1, df2)
        not_switch, switch = 0, 0
        for row_idx in range(df1.shape[0]):
            if (perm_df1.iloc[row_idx].equals(perm_df1.iloc[row_idx])
                    and perm_df2.iloc[row_idx].equals(perm_df2.iloc[row_idx])):
                not_switch += 1
            if (perm_df1.iloc[row_idx].equals(perm_df2.iloc[row_idx])
                    and perm_df2.iloc[row_idx].equals(perm_df1.iloc[row_idx])):
                switch += 1
        self.assertEqual(switch + not_switch, df1.shape[0])

    def test_get_rms(self):
        Y = [1, 3, 4, 5, 2, 3, 4]
        X = range(1, 8)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        rms = get_rms(model, X, Y)
        self.assertTrue(np.isclose(rms, 1.1406228159050935))

    def test_get_permutation_stats(self):
        np.random.seed(0)
        df1 = pd.DataFrame({
            'Year_Month': ['{:4d}-{:02d}'.format(year, month) for year in range(1980, 2010) for month in range(1, 13)],
            'Total_Crime_Count': np.random.randint(500, size=360),
            '0-10': np.random.randint(5, size=360),
            '10-20': np.random.randint(5, size=360),
            '20-30': np.random.randint(5, size=360)
        }).set_index('Year_Month')
        df2 = pd.DataFrame({
            'Year_Month': ['{:4d}-{:02d}'.format(year, month) for year in range(1980, 2010) for month in range(1, 13)],
            'Total_Crime_Count': np.random.randint(500, size=360),
            '0-10': np.random.randint(5, size=360),
            '10-20': np.random.randint(5, size=360),
            '20-30': np.random.randint(5, size=360)
        }).set_index('Year_Month')
        stats = get_permutation_stats(df1, df2, 'Total_Crime_Count', 10)
        self.assertEqual(len(stats), 10)
        for stat in stats:
            self.assertTrue(stat >= 0)
            self.assertIsInstance(stat, float)


if __name__ == '__main__':
    unittest.main()
