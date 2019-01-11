import unittest
from code import *

print("Preparing test data...")
nan = float('nan')

sample_stations = [{'ID': 'USC00046336',
                    'LATITUDE': 37.8123,
                    'LONGITUDE': -122.21600000000001,
                    'ELEVATION': 113.4,
                    'STATE': 'CA',
                    'NAME': 'PIEDMONT 1.0 SE',
                    'GSN FLAG': nan,
                    'HCN/CRN FLAG': nan,
                    'WMO ID': nan},
                   {'ID': 'USC00040693',
                    'LATITUDE': 37.7075,
                    'LONGITUDE': -122.0687,
                    'ELEVATION': 87.5,
                    'STATE': 'CA',
                    'NAME': 'CASTRO VALLEY 0.5 WSW',
                    'GSN FLAG': nan,
                    'HCN/CRN FLAG': nan,
                    'WMO ID': nan}]

weather_data = load_weather_data(sample_stations, start_year=1980, end_year=2009)
temperature_data = get_temperature_data(weather_data)
precipitation_data = get_precipitation_data(weather_data)

temperature_bias = compute_station_bias(temperature_data)
precipitation_bias = compute_station_bias(precipitation_data)

temperature_data_wbias = apply_station_bias(temperature_data, temperature_bias)
precipitation_data_wbias = apply_station_bias(precipitation_data, precipitation_bias)

sample_distances = [0.07466120567716328, 0.10640821245750016, ]
station_weights = get_station_weights(sample_stations, sample_distances)


class TestFunctions(unittest.TestCase):

    def test_download_and_cache_file(self):
        """
        Test download_and_cache_file() function
        - Download and check fo the file
        """
        path = download_and_cache_file("stations_ca.csv", "https://osf.io/ev7y2/download",
                                       force_download=True)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.basename(path), "stations_ca.csv")

    def test_filter_weather_data(self):
        columns = ["ID", "YEARMONTHDAY", "ELEMENT", "DATA VALUE",
                   "M-FLAG", "Q-FLAG", "S-FLAG", "OBS-TIME"]
        filtered = filter_weather_data(weather_data[:10])
        self.assertEqual(len(filtered), 10)
        for elem in filtered:
            self.assertIsInstance(elem, dict)
            for column in columns:
                self.assertIn(column, elem)

        bad_data = [{
            "ID": 1, "YEARMONTHDAY": 20000101, "ELEMENT": "TMAX", "DATA VALUE": 0,
            "M-FLAG": 0, "Q-FLAG": 0, "S-FLAG": 0, "OBS-TIME": 0
        },
        {
            "ID": 1, "YEARMONTHDAY": 20000101, "ELEMENT": "TMIN", "DATA VALUE": 100,
            "M-FLAG": 0, "Q-FLAG": 0, "S-FLAG": 0, "OBS-TIME": 0
        }]
        filtered = filter_weather_data(bad_data)
        self.assertEqual(len(filtered), 0)

        no_min_max = [{
            "ID": 1, "YEARMONTHDAY": 20000101, "ELEMENT": "TMAX", "DATA VALUE": 0,
            "M-FLAG": 0, "Q-FLAG": 0, "S-FLAG": 0, "OBS-TIME": 0
        },
        {
            "ID": 1, "YEARMONTHDAY": 20000102, "ELEMENT": "TMIN", "DATA VALUE": 0,
            "M-FLAG": 0, "Q-FLAG": 0, "S-FLAG": 0, "OBS-TIME": 0
        }]
        filtered = filter_weather_data(no_min_max)
        self.assertEqual(len(filtered), 2)

    def test_get_temperature_data(self):
        """
        Test get_temperature_data(weather_data) function
        - Check the return value should be list
        - Check the return value (list) is composed of dictionary elements
        - Check the dict element contains three keys
        - Check the DATA VALUE is within the reasonable range that is to check it is Fahrenheit
        """

        self.assertEqual(type(temperature_data), list,
                         "Type of result should be list")
        self.assertEqual(type(temperature_data[0]), dict,
                         "Type of result should be dictionary")
        self.assertEqual(len(temperature_data[0].keys()), 3)

        self.assertGreater(temperature_data[0]['DATA VALUE'], -40)
        self.assertLess(temperature_data[0]['DATA VALUE'], 150)

    def test_get_precipitation_data(self):
        """
        Test get_precipitation_data(weather_data) function
        - Check the return value should be list
        - Check the return value (list) is composed of dictionary elements
        - Check the dict element contains three keys
        - Check the DATA VALUE is within the reasonable range that is to check it is Fahrenheit
        """
        self.assertEqual(type(precipitation_data), list,
                         "Type of result should be list")
        self.assertEqual(type(precipitation_data[0]), dict,
                         "Type of result should be dictionary")
        self.assertEqual(len(precipitation_data[0].keys()), 3)

        self.assertGreater(precipitation_data[0]['DATA VALUE'], -1)
        self.assertLess(precipitation_data[0]['DATA VALUE'], 200)

    def test_find_valid_stations_id_each_year(self):
        """
        Test get_county_rectangle_border(county) function
        - Check the return value should be a dictionary
        - Check the return dictionary should have 'Year' as key
        - Check the return dictionary should have Station Id as value
        """
        valid_stations = find_valid_stations_id_each_year(weather_data)
        self.assertEqual(type(valid_stations), dict,
                         "Type of result should be dictionary")
        year, station_id_set = valid_stations.popitem()
        station_id = station_id_set.pop()
        self.assertEqual(len(str(year)), 4,
                         "Length of result should be 4")
        self.assertEqual(len(station_id), 11,
                         "Length of result should be 11")
        self.assertEqual(station_id[:2], 'US')

    def test_compute_station_bias(self):
        """
        Test compute_station_bias(weather_data) function
        - Check the return value should be dictionary
        - Check the return dict value (bias) should be in a reasonable range
        - Check the return dict key (station_id) is in US format and has a length of 11
        - Check that again with a different input
        """
        self.assertEqual(type(temperature_bias), dict,
                         "Type of result should be dictionary")
        station_id, bias = temperature_bias.popitem()
        self.assertGreater(bias, -10)
        self.assertLess(bias, 10)
        self.assertEqual(station_id[:2], 'US')
        self.assertEqual(len(station_id), 11,
                         "Length of result should be 11")

        self.assertEqual(type(precipitation_bias), dict,
                         "Type of result should be dictionary")
        station_id, bias = precipitation_bias.popitem()
        self.assertGreater(bias, -3)
        self.assertLess(bias, 3)
        self.assertEqual(station_id[:2], 'US')
        self.assertEqual(len(station_id), 11,
                         "Length of result should be 11")

    def test_apply_station_bias(self):
        """
        Test apply_station_bias(weather_data, bias) function
        - Check the return value should be list
        - Check the return value (list) is composed of dictionary elements
        - Check the dict element contains three keys
        - Check the DATA VALUE is within the reasonable range that is to check it is Fahrenheit
        - Check that again with precipitation data
        """

        self.assertEqual(type(temperature_data_wbias), list,
                         "Type of result should be list")
        self.assertEqual(type(temperature_data_wbias[0]), dict,
                         "Type of result should be dictionary")
        self.assertEqual(len(temperature_data_wbias[0].keys()), 3)

        self.assertGreater(temperature_data_wbias[0]['DATA VALUE'], -40)
        self.assertLess(temperature_data_wbias[0]['DATA VALUE'], 150)

        self.assertEqual(type(precipitation_data_wbias), list,
                         "Type of result should be list")
        self.assertEqual(type(precipitation_data_wbias[0]), dict,
                         "Type of result should be dictionary")
        self.assertEqual(len(precipitation_data_wbias[0].keys()), 3)

        self.assertGreater(precipitation_data_wbias[0]['DATA VALUE'], -1)
        self.assertLess(precipitation_data_wbias[0]['DATA VALUE'], 200)

    def test_get_station_weights(self):
        """
        Test get_station_weights(stations, distances)
        - Check the return value should be dict
        - Check the return dict value (bias) should be in a reasonable range
        - Check the return dict key (station_id) is in US format and has a length of 11
        """

        self.assertEqual(type(station_weights), dict,
                         "Type of result should be dictionary")
        station_id, weights = station_weights.popitem()
        self.assertGreater(weights, 0)
        self.assertLess(weights, 1)
        self.assertEqual(station_id[:2], 'US')
        self.assertEqual(len(station_id), 11,
                         "Length of result should be 11")

    def test_aggregate_temperature_data(self):
        """
        Test aggregate_temperature_data(temperature_data, station_weights) function
        - Check the return value should be list
        - Check the element in return list should be list
        - Check the count in bin should be in reasonable range
        - Check the return value should have a length of 11 due to 11 bins created
        - Check the element in return list should have a length of around 360 (sample station)
        """
        bins_count = aggregate_temperature_data(temperature_data, station_weights)
        self.assertEqual(type(bins_count), list,
                         "Type of result should be list")
        self.assertEqual(type(bins_count[0]), list,
                         "Type of result should be list")

        self.assertGreater(bins_count[0][0], -1)
        self.assertLess(bins_count[0][0], 32)

        self.assertEqual(len(bins_count), 11,
                         "Length of result should be 11")
        self.assertGreater(len(bins_count[0]), 300)

    def test_aggregate_precipitation_data(self):
        """
        Test aggregate_precipitation_data(precipitation_data, station_weights) function
        - Check the return value should be list
        - Check the element in return list should be list
        - Check the count in bin should be in reasonable range
        - Check the return value should have a length of 11 due to 11 bins created
        - Check the element in return list should have a length of around 360 (sample station)
        """
        bins_count = aggregate_precipitation_data(precipitation_data, station_weights)
        self.assertEqual(type(bins_count), list,
                         "Type of result should be list")
        self.assertEqual(type(bins_count[0]), list,
                         "Type of result should be list")

        self.assertGreater(bins_count[0][0], -1)
        self.assertLess(bins_count[0][0], 32)

        self.assertEqual(len(bins_count), 5,
                         "Length of result should be 11")
        self.assertGreater(len(bins_count[0]), 300)


if __name__ == '__main__':
    unittest.main()
