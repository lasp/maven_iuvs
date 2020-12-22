# Built-in imports
from unittest import TestCase
from datetime import date, timedelta
import warnings

# 3rd-party imports
import numpy as np

# Local imports
from PyUVS.time import ScienceWeek


class TestScienceWeek(TestCase):
    def setUp(self):
        self.science_week = ScienceWeek()


class TestScienceWeekInit(TestScienceWeek):
    def test_science_week_has_science_start_date_attribute(self):
        self.assertTrue(hasattr(self.science_week, 'science_start_date'))

    def test_science_week_has_exactly_one_parameter(self):
        self.assertTrue(len(self.science_week.__dict__.keys()), 1)

    def test_science_start_date_is_2014_11_11(self):
        self.assertEqual(date(2014, 11, 11), self.science_week.science_start_date)

    def test_science_start_date_cannot_be_modified(self):
        with self.assertRaises(AttributeError):
            self.science_week.science_start_date = date(2014, 11, 11)


class TestGetScienceWeekFromDate(TestScienceWeek):
    def test_science_start_date_is_week_0(self):
        self.assertEqual(0, self.science_week.get_science_week_from_date(self.science_week.science_start_date))

    def test_known_science_week_and_date_matches_output(self):
        test_date = date(2020, 12, 14)
        self.assertEqual(317, self.science_week.get_science_week_from_date(test_date))

    def test_integer_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.science_week.get_science_week_from_date(100)


# I cannot figure out how to properly mock a date such that ScienceWeek knows about my mock
class TestGetCurrentScienceWeek(TestScienceWeek):
    # TODO: Get a test for this
    pass
    '''def test_science_week_of_today(self):
        with mock.patch('maven_iuvs.science_week.science_week.datetime.date') as mock_date:
            mock_date.today.return_value = date(2020, 1, 1)
            #self.assertEqual(mock_date.today(), date(2020, 1, 1))   # this works
            self.assertEqual(268, self.science_week.get_current_science_week())  # This doesn't about mock_date'''


class TestGetScienceWeekStartDate(TestScienceWeek):
    def test_start_of_week_0_is_mission_arrival_date(self):
        self.assertEqual(self.science_week.science_start_date, self.science_week.get_science_week_start_date(0))

    def test_example_science_week_matches_its_known_start_date(self):
        self.assertEqual(date(2020, 12, 8), self.science_week.get_science_week_start_date(317))

    def test_floats_between_two_consecutive_integers_give_output_equal_to_lower_integer(self):
        with warnings.catch_warnings(record=True):
            self.assertEqual(self.science_week.get_science_week_start_date(317),
                             self.science_week.get_science_week_start_date(317.0),
                             self.science_week.get_science_week_start_date(np.nextafter(318, 317)))

    def test_integer_float_week_raises_no_warnings(self):
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            self.science_week.get_science_week_start_date(1.0)
            self.assertEqual(0, len(warning))

    def test_non_integer_float_raises_warning(self):
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            self.science_week.get_science_week_start_date(1.5)
            self.assertEqual(1, len(warning))
            self.assertEqual(warning[-1].category, UserWarning)

    def test_ndarray_input_raises_type_error(self):
        test_weeks = np.linspace(1, 50, num=50, dtype=int)
        with self.assertRaises(TypeError):
            self.science_week.get_science_week_start_date(test_weeks)

    def test_string_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.science_week.get_science_week_start_date('100')

    def test_nan_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.science_week.get_science_week_start_date(np.nan)


class TestGetScienceWeekEndDate(TestScienceWeek):
    def test_end_of_week_0_is_6_days_after_science_start_date(self):
        self.assertEqual(self.science_week.science_start_date + timedelta(days=6),
                         self.science_week.get_science_week_end_date(0))

    def test_example_science_week_matches_its_known_end_date(self):
        self.assertEqual(date(2020, 12, 14), self.science_week.get_science_week_end_date(317))

    def test_floats_between_two_consecutive_integers_give_output_equal_to_lower_integer(self):
        with warnings.catch_warnings(record=True):
            self.assertEqual(self.science_week.get_science_week_end_date(317),
                             self.science_week.get_science_week_end_date(317.0),
                             self.science_week.get_science_week_end_date(np.nextafter(318, 317)))

    def test_integer_float_week_raises_no_warnings(self):
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            self.science_week.get_science_week_end_date(1.0)
            self.assertEqual(0, len(warning))

    def test_non_integer_float_raises_warning(self):
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            self.science_week.get_science_week_end_date(1.5)
            self.assertEqual(1, len(warning))
            self.assertEqual(warning[-1].category, UserWarning)

    def test_ndarray_input_raises_type_error(self):
        test_weeks = np.linspace(1, 50, num=50, dtype=int)
        with self.assertRaises(TypeError):
            self.science_week.get_science_week_end_date(test_weeks)

    def test_string_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.science_week.get_science_week_end_date('100')

    def test_nan_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.science_week.get_science_week_end_date(np.nan)


class TestGetScienceWeekDateRange(TestScienceWeek):
    def test_output_is_tuple(self):
        self.assertTrue(isinstance(self.science_week.get_science_week_date_range(100), tuple))

    def test_output_is_2_elements(self):
        self.assertTrue(len(self.science_week.get_science_week_date_range(100)), 2)

    def test_output_is_result_of_start_and_end_date_methods(self):
        self.assertTrue(self.science_week.get_science_week_start_date(100),
                        self.science_week.get_science_week_date_range(100)[0])
        self.assertTrue(self.science_week.get_science_week_end_date(100),
                        self.science_week.get_science_week_date_range(100)[1])
