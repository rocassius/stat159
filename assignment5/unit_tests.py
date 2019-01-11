import math
import unittest

from code import *

nan = float('nan')


def square_list(l):
    return list(map(lambda x: x ** 2, l))


def double_list(l):
    return list(map(lambda x: x * 2, l))


def add_five_list(l):
    return list(map(lambda x: x + 5, l))


class TestFunctions(unittest.TestCase):
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

    def test_bin_temperature_data(self):

        temperature_data = [
            {'DATA VALUE': 59.0, 'ID': 'USC00040693', 'YEARMONTHDAY': 19800101},
            {'DATA VALUE': 62.959999999999994, 'ID': 'USC00040693', 'YEARMONTHDAY': 19800229},
            {'DATA VALUE': 77.0, 'ID': 'USC00040693', 'YEARMONTHDAY': 19800411},
            {'DATA VALUE': 66.6, 'ID': 'USC99999999', 'YEARMONTHDAY': 19800411},
        ]

        result_dict = bin_temperature_data(temperature_data)

        self.assertEqual(type(result_dict), dict,
                         "Type of result should be dict")

        self.assertEqual(type(result_dict['USC00040693']), dict,
                         "Type of result should be dict")

        self.assertEqual(result_dict['USC00040693'],
                         {19800101: '50-60', 19800229: '60-70', 19800411: '70-80'})

        self.assertTrue(math.isnan(result_dict['USC99999999'][19800101]))
        self.assertTrue(math.isnan(result_dict['USC99999999'][19800229]))
        self.assertEqual(result_dict['USC99999999'][19800411], '60-70')

    def test_remove_na_from_data(self):

        bined_temperature_data_1 = {
            19800101: '50-60',
            19800229: '60-70',
            19800411: '70-80'}
        bined_temperature_data_2 = {
            19800101: nan,
            19800229: nan,
            19800411: '60-70'}

        clean_result1, clean_result2 = remove_na_from_data(bined_temperature_data_1,
                                                           bined_temperature_data_2)
        self.assertEqual(type(clean_result1), dict,
                         "Type of result should be dict")
        self.assertEqual(type(clean_result2), dict,
                         "Type of result should be dict")
        self.assertEqual(len(clean_result1), len(clean_result2))
        self.assertEqual(clean_result1.keys(), clean_result2.keys())
        self.assertEqual(clean_result1[19800411], '70-80')
        self.assertEqual(clean_result2[19800411], '60-70')

    def test_get_weather_permutation(self):
        clean_temperature_data_1 = {19800411: '70-80'}
        clean_temperature_data_2 = {19800411: '60-70'}
        swap_result_1, swap_result_2 = get_weather_permutation(clean_temperature_data_1,
                                                               clean_temperature_data_2)

        self.assertEqual(type(swap_result_1), dict,
                         "Type of result should be dict")
        self.assertEqual(type(swap_result_1), dict,
                         "Type of result should be dict")

        self.assertTrue((swap_result_1[19800411] == '70-80' and swap_result_2[19800411] == '60-70') or
                        (swap_result_1[19800411] == '60-70' and swap_result_2[19800411] == '70-80'))

    def test_aggregate_bin_temperature_by_month(self):

        temperature_data = {19800101: '50-60',
                            19800129: '50-60',
                            19800211: '70-80'}

        agg_result = aggregate_bin_temperature_by_month(temperature_data)

        self.assertEqual(type(agg_result), dict,
                         "Type of result should be dict")
        self.assertEqual(type(agg_result[198001]), dict,
                         "Type of result should be dict")

        for temp_bin in TEMPERATURE_BINS_LABEL:
            if temp_bin == '50-60':
                self.assertEqual(agg_result[198001][temp_bin], 2)
            else:
                self.assertEqual(agg_result[198001][temp_bin], 0)
        for temp_bin in TEMPERATURE_BINS_LABEL:
            if temp_bin == '70-80':
                self.assertEqual(agg_result[198002][temp_bin], 1)
            else:
                self.assertEqual(agg_result[198002][temp_bin], 0)

    def test_compute_p_values(self):

        count_dict_1 = {198001: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 0,
                                 '50-60': 2, '60-70': 0, '70-80': 0, '80-90': 0, '90-100': 0, '100-110': 0},
                        198002: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 9,
                                 '50-60': 2, '60-70': 0, '70-80': 0, '80-90': 0, '90-100': 0, '100-110': 0}}
        count_dict_2 = {198001: {'0-10': 0, '10-20': 0, '20-30': 0, '30-40': 0, '40-50': 0,
                                 '50-60': 0, '60-70': 0, '70-80': 1, '80-90': 0, '90-100': 0, '100-110': 0}}

        p_vals = compute_p_values(count_dict_1, count_dict_2)

        self.assertEqual(type(p_vals), list,
                         "Type of result should be list")
        self.assertEqual(len(p_vals), 1)
        self.assertEqual(p_vals[0], 0.66500554210202911)

    def test_eliminate_zero_expectations(self):

        A = [[1, 4, 5, 0], [-5, 8, 9, 0], [-6, 7, 11, 0]]
        B = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        new_a = eliminate_zero_expectations(A)
        new_b = eliminate_zero_expectations(B)

        self.assertEqual(type(new_a), np.ndarray,
                         "Type of result should be np.ndarray")
        self.assertEqual(len(new_a[0]), 3,
                         "The row which is 0s should be eliminated")
        self.assertEqual(type(new_b), np.ndarray,
                         "Type of result should be np.ndarray")
        self.assertEqual(len(new_b), 0,
                         "The matrix should now be empty")

    def test_fisher_combine(self):

        probs = [0.8, 0.2, 0.5]
        result = fisher_combine(probs)

        self.assertEqual(type(result), np.float64,
                         "Type of result should be float")

    def test_permutation_test(self):
        station1 = {19800101: '50-60', 19800102: '50-60', 19800103: '50-60',
                    19800104: '50-60', 19800105: '50-60', 19800106: '50-60',
                    19800107: '60-70', 19800108: '50-60', 19800109: '50-60'}
        station2 = {19800101: '10-20', 19800102: '20-30', 19800103: '40-50',
                    19800104: '30-40', 19800105: '50-60', 19800106: '80-90',
                    19800107: '60-70', 19800108: '10-20', 19800109: '70-80'}

        result = permutation_test(station1, station2, reps=200)
        self.assertEqual(type(result), list,
                         "Type of result should be list")
        self.assertEqual(len(result), 200,
                         "List should have a length of 200")
        self.assertNotEqual(result[:100], result[100:200])

    def test_permutation_test_executor(self):
        station1 = {19800101: '50-60', 19800102: '50-60', 19800103: '50-60',
                    19800104: '50-60', 19800105: '50-60', 19800106: '50-60',
                    19800107: '60-70', 19800108: '50-60', 19800109: '50-60'}
        station2 = {19800101: '10-20', 19800102: '20-30', 19800103: '40-50',
                    19800104: '30-40', 19800105: '50-60', 19800106: '80-90',
                    19800107: '60-70', 19800108: '10-20', 19800109: '70-80'}

        rep_range = [0 for _ in range(100)]
        result = permutation_test_executor(rep_range, station1, station2)

        self.assertEqual(type(result), list,
                         "Type of result should be list")
        self.assertEqual(len(result), 100,
                         "List should have a length of 200")
        self.assertNotEqual(result[:50], result[50:100])

    def test_permutation_test_executor_parallel(self):
        station1 = {19800101: '50-60', 19800102: '50-60', 19800103: '50-60',
                    19800104: '50-60', 19800105: '50-60', 19800106: '50-60',
                    19800107: '60-70', 19800108: '50-60', 19800109: '50-60'}
        station2 = {19800101: '10-20', 19800102: '20-30', 19800103: '40-50',
                    19800104: '30-40', 19800105: '50-60', 19800106: '80-90',
                    19800107: '60-70', 19800108: '10-20', 19800109: '70-80'}

        rep_range = [0 for _ in range(100)]
        result = permutation_test_executor_parallel(rep_range, station1, station2)

        self.assertEqual(type(result), list,
                         "Type of result should be list")
        self.assertEqual(len(result), 100,
                         "List should have a length of 200")
        self.assertNotEqual(result[:50], result[50:100])


if __name__ == '__main__':
    unittest.main()
