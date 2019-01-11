import unittest
import numpy
from code import *


class TestFunctions(unittest.TestCase):

    def test_generate_random_grid(self):
        """
        Test generate_random_grid() function
        - Check the return value should be list
        - Check the length of the list should larger than 0
        - Check that each element of the list is a tuple
        - Check each tuple has length of 2
        - Check the data type inside tuple should be float
        """
        np.random.seed(0)
        grid = generate_random_grid()

        self.assertEqual(type(grid), list,
                         "Type of grid should be list")
        self.assertEqual(len(grid) > 0, True,
                         "Length of grid should be larger than 0")
        self.assertEqual(type(grid[0]), tuple,
                         "Type of each point should be tuple")
        self.assertEqual(len(set(map(type, grid))), 1,
                         "Type of each point should be tuple")
        self.assertEqual(len(grid[0]), 2,
                         "Length of each point tuple should be 2")
        self.assertEqual(len(set(map(len, grid))), 1,
                         "Length of each point tuple should be 2")
        self.assertEqual(type(grid[0][0]), numpy.float64,
                         "Type of the point representing Lat should be float")
        self.assertEqual(type(grid[0][1]), numpy.float64,
                         "Type of the point representing Lon should be float")

    def test_generate_random_stations(self):
        """
        Test generate_random_stations()
        - Check the return value should be list
        - Check the length of the list should match the specified value
        - Check that each element of the list is a tuple
        - Check each tuple has length of 2
        - Check the function can take edge case input (1)
        """
        np.random.seed(0)
        stations = generate_random_stations(15)

        self.assertEqual(type(stations), list,
                         "Type of stations should be list")
        self.assertEqual(len(stations), 15,
                         "Length of stations should be 15")
        self.assertEqual(type(stations[0]), tuple,
                         "Type of each point should be tuple")
        self.assertEqual(len(set(map(type, stations))), 1,
                         "Type of each point should be tuple")
        self.assertEqual(len(stations[0]), 2,
                         "Length of each point tuple should be 2")
        self.assertEqual(len(set(map(len, stations))), 1,
                         "Length of each point tuple should be 2")

        station2 = generate_random_stations(1)

        self.assertEqual(type(station2), list,
                         "Type of stations should be list")
        self.assertEqual(len(station2), 1,
                         "Length of stations should be 1")
        self.assertEqual(type(station2[0]), tuple,
                         "Type of each point should be tuple")
        self.assertEqual(len(set(map(type, station2))), 1,
                         "Type of each point should be tuple")
        self.assertEqual(len(station2[0]), 2,
                         "Length of each point tuple should be 2")
        self.assertEqual(len(set(map(len, station2))), 1,
                         "Length of each point tuple should be 2")

    def test_compute_average_inverse_distance(self):
        """
        Test compute_average_inverse_distance()
        - Check the return value should be list
        - Check the length of the list should match the number of stations
        - Check that each element of the list is a float
        """
        np.random.seed(0)
        grid = generate_random_grid()
        stations = generate_random_stations(15)
        distances = compute_average_inverse_distance(grid, stations)

        self.assertEqual(type(distances), list,
                         "Type of stations should be list")
        self.assertEqual(len(distances), len(stations),
                         "Length of stations should be 15")
        self.assertEqual(type(distances[0]), float,
                         "Type of each distance should be float")
        self.assertEqual(len(set(map(type, distances))), 1,
                         "Type of each distance should be float")


if __name__ == '__main__':
    unittest.main()
