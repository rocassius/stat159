import unittest

from code import *

EAST_ZIPCODES = [94568, 94588, 94566, 94550, 94586, 94551, 94514]

class TestFunctions(unittest.TestCase):

	def test_check_coordinate_within_zipcodes(self):
		west_uc_berkeley = [37.8499435, -122.255239]
		east_livermore_lab = [37.6689487, -121.834272]

		(self.assertFalse(
		 	check_coordinate_within_zipcodes(
		 		west_uc_berkeley, EAST_ZIPCODES)))

		(self.assertTrue(
		 	check_coordinate_within_zipcodes(
		 		east_livermore_lab, EAST_ZIPCODES)))

	def test_split_grid(self):
		grid = [[37.8499435, -122.255239],
				[37.6689487, -121.834272],
				[37.7141047, -121.81282]]
		east_grid, west_grid = split_grid(grid, EAST_ZIPCODES)

		self.assertEqual(east_grid, [[37.6689487, -121.834272],
				[37.7141047, -121.81282]])
		self.assertEqual(west_grid, [[37.8499435, -122.255239]])









if __name__ == '__main__':
    unittest.main()
