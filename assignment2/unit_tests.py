import unittest
from code import *


def square_list(l):
    return list(map(lambda x: x ** 2, l))


def double_list(l):
    return list(map(lambda x: x * 2, l))


def add_five_list(l):
    return list(map(lambda x: x + 5, l))


class TestFunctions(unittest.TestCase):

    def test_find_zero(self):
        """
        Test find_zero() function
        - Check the return value should be float
        - Check the return value plugged into the function gives zero (or close enough) as expected
        - Check that with different functions
        """
        result1 = find_zero(lambda delta_lon:
                            distance((37.905098, -122.272225), (37.905098, -122.272225 + delta_lon)).miles - 10,
                            guess=0.09)
        self.assertEqual(type(result1), float,
                         "Type of result should be float")
        self.assertAlmostEqual(
            round((distance((37.905098, -122.272225), (37.905098, -122.272225 + result1)).miles - 10), 2), 0)

        result2 = find_zero(lambda x: 2.5 * x ** 2 + 3 * x - 10, guess=2)

        self.assertEqual(type(result2), float,
                         "Type of result should be float")
        self.assertAlmostEqual(round(2.5 * result2 ** 2 + 3 * result2 - 10, 2), 0)

    def test_parallel_list_map(self):
        """
        Test parallel_list_map() function
        - Check if the function produce the same result as run without multi-threading
        """
        lst = [1, 5, 3, 5, 7, 3, 5, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8]

        self.assertEqual(square_list(lst), parallel_list_map(lst, square_list),
                         "Result not match for square_list")
        self.assertEqual(double_list(lst), parallel_list_map(lst, double_list),
                         "Result not match for double_list")
        self.assertEqual(add_five_list(lst), parallel_list_map(lst, add_five_list),
                         "Result not match for add_five_list")

    def test_download_and_cache_file(self):
        """
        Test download_and_cache_file() function
        - Download and check fo the file
        """
        path = download_and_cache_file("stations_ca.csv", "https://osf.io/ev7y2/download",
                                       force_download=True)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.basename(path), "stations_ca.csv")

    def test_check_coordinate_within_county(self):
        """
        Test check_coordinate_within_county() function
        - Check Berkeley within Alameda
        - Check Mountain View out of Alameda
        """
        geo_locator = Nominatim(user_agent=USER_AGENT)

        location = geo_locator.geocode("Berkeley")
        self.assertTrue(check_coordinate_within_county((location.latitude, location.longitude), "Alameda County"))

        location = geo_locator.geocode("Mountain View")
        self.assertFalse(check_coordinate_within_county((location.latitude, location.longitude), "Alameda County"))

    def test_get_county_rectangle_border(self):
        """
        Test get_county_rectangle_border(county) function
        - Check the return value should be list
        - Check the return value should be length of 2
        - Check the element in return list should be tuple
        - Check the element in return list should be length of 2 since it is a coordinate
        """
        border = get_county_rectangle_border("Alameda County")
        self.assertEqual(type(border), list,
                         "Type of result should be list")
        self.assertEqual(type(border[0]), tuple,
                         "Type of result should be tuple")
        self.assertEqual(len(border[0]), 2,
                         "Type of result should be lat and lon")
        self.assertEqual(len(border), 2,
                         "Type of result should be two coordinates")

    def test_get_county_polygon_border(self):
        """
        Test get_county_rectangle_border(county) function
        - Check the return value should be list
        - Check the element in return list should be tuple
        - Check the element in return list should be length of 2 since it is a coordinate
        """
        border = get_county_polygon_border("Alameda County")
        self.assertEqual(type(border), list,
                         "Type of result should be list")
        self.assertEqual(type(border[0]), tuple,
                         "Type of result should be tuple")
        self.assertEqual(len(border[0]), 2,
                         "Type of result should be lat and lon")

    def test_generate_grid_within_rectangle_bounds(self):
        """
        Test get_county_rectangle_border(county) function
        - Check the return value should be list
        - Check the element in return list should be tuple
        - Check the element in return list should be length of 2 since it is a coordinate
        - Check it is a grid not just a border (>2)
        """
        grid_raw = generate_grid_within_rectangle_bounds("Alameda County",
                                                         start_point=(37.905098, -122.272225),
                                                         spacing=10)
        self.assertEqual(type(grid_raw), list,
                         "Type of result should be list")
        self.assertEqual(type(grid_raw[0]), tuple,
                         "Type of result should be tuple")
        self.assertEqual(len(grid_raw[0]), 2,
                         "Type of result should be lat and lon")
        self.assertGreater(len(grid_raw), 2,
                           "Type of result should be lat and lon")

        grid = filter_grid(grid_raw, "Alameda County")

        self.assertEqual(type(grid), list,
                         "Type of result should be list")
        self.assertEqual(type(grid[0]), tuple,
                         "Type of result should be tuple")
        self.assertEqual(len(grid[0]), 2,
                         "Type of result should be lat and lon")
        self.assertGreater(len(grid), 2,
                           "Type of result should be lat and lon")

        grid = filter_grid_parallel(grid_raw, "Alameda County")

        self.assertEqual(type(grid), list,
                         "Type of result should be list")
        self.assertEqual(type(grid[0]), tuple,
                         "Type of result should be tuple")
        self.assertEqual(len(grid[0]), 2,
                         "Type of result should be lat and lon")
        self.assertGreater(len(grid), 2,
                           "Type of result should be lat and lon")

    def test_grid_includes_starting_point(self):
        grid = generate_grid_within_rectangle_bounds("Alameda County", start_point=(37.905098, -122.272225))
        self.assertTrue((37.905098, -122.272225) in grid)

    def test_location_in_boundary(self):
        # coordinates of UC Berkeley and Soda hall.
        uc_berkeley_coord = {'LATITUDE': 37.8719034, 'LONGITUDE': -122.2607339}
        soda_hall_coord = {'LATITUDE': 37.8755981, 'LONGITUDE': -122.2609805}

        stations = [uc_berkeley_coord, soda_hall_coord]
        filtered_stations = filter_stations_ec(stations, "Alameda")

        self.assertEqual(len(filtered_stations), 2)
        self.assertEqual(filtered_stations, [uc_berkeley_coord, soda_hall_coord])

        filtered_stations_parallel = filter_stations_ec_parallel(stations, "Alameda")

        self.assertEqual(len(filtered_stations_parallel), 2)
        self.assertEqual(filtered_stations_parallel, [uc_berkeley_coord, soda_hall_coord])

    def test_location_out_boundary_in_max_distance(self):
        # coordinates of San Francisco city hall.
        city_hall_coord = {'LATITUDE': 37.7792597, 'LONGITUDE': -122.4280193}
        mt_diablo = {'LATITUDE': 37.883877, 'LONGITUDE': -121.917303}

        stations = [city_hall_coord, mt_diablo]
        filtered_stations = filter_stations_ec(stations, "Alameda")

        self.assertEqual(len(filtered_stations), 1)
        self.assertEqual(filtered_stations, [city_hall_coord])

        filtered_stations_parallel = filter_stations_ec_parallel(stations, "Alameda")

        self.assertEqual(len(filtered_stations_parallel), 1)
        self.assertEqual(filtered_stations_parallel, [city_hall_coord])

    def test_location_out_boundary_out_max_distance(self):
        # coordinates of ucla
        ucla_coord = {'LATITUDE': 34.0689254, 'LONGITUDE': -118.4473751}

        stations = [ucla_coord]
        filtered_stations = filter_stations_ec(stations, "Alameda")

        self.assertEqual(len(filtered_stations), 0)
        self.assertEqual(filtered_stations, [])

        filtered_stations_parallel = filter_stations_ec_parallel(stations, "Alameda")

        self.assertEqual(len(filtered_stations_parallel), 0)
        self.assertEqual(filtered_stations_parallel, [])

    def test_load_stations(self):
        """
        Test load_stations() function
        """
        # Load stations from the file
        stations = load_stations()
        self.assertIsInstance(stations, list)
        for station in stations:
            self.assertIsInstance(station, dict)
            self.assertTrue('LATITUDE' in station)
            self.assertTrue('LONGITUDE' in station)

        coordinates = convert_stations_to_coordinates(stations)
        self.assertIsInstance(stations, list)
        for coordinate in coordinates:
            self.assertIsInstance(coordinate, tuple)
            self.assertIsInstance(coordinate[0], float)
            self.assertIsInstance(coordinate[1], float)

    def test_filter_stations(self):
        grid = generate_grid_within_rectangle_bounds("Alameda County", start_point=(37.905098, -122.272225))
        stations = load_stations()

        stations = filter_stations_parallel(stations, grid)
        for station in stations:
            self.assertIsInstance(station, dict)
            self.assertTrue('LATITUDE' in station)
            self.assertTrue('LONGITUDE' in station)


if __name__ == '__main__':
    unittest.main()
