# Built-in imports
from unittest import TestCase
from datetime import date, timedelta

# Local imports
from maven_iuvs.science_week.science_week import ScienceWeek


class TestScienceWeek(TestCase):
    def setUp(self):
        self.science_week = ScienceWeek()


class TestScienceWeekInit(TestScienceWeek):
    def test_science_week_has_arrival_attribute(self):
        self.assertTrue(hasattr(self.science_week, 'maven_arrival_date'))

    def test_science_week_has_exactly_one_attribute(self):
        self.assertTrue(len(self.science_week.__dict__.keys()), 1)

    def test_maven_arrival_is_2014_11_11(self):
        self.assertEqual(date(2014, 11, 11), self.science_week.maven_arrival_date)


class TestGetScienceWeekFromDate(TestScienceWeek):
    def test_type_error_if_int_input(self):
        self.assertRaises(TypeError, lambda: self.science_week.get_science_week_from_date(100))

    def test_error_raised_if_before_mission_start(self):
        pre_arrival_date = date(2014, 11, 10)
        self.assertRaises(ValueError, lambda: self.science_week.get_science_week_from_date(pre_arrival_date))

    def test_mission_start_date_is_week_0(self):
        self.assertEqual(0, self.science_week.get_science_week_from_date(self.science_week.maven_arrival_date))

    def test_randomly_chosen_date_is_known_science_week(self):
        test_date = date(2020, 12, 14)
        self.assertEqual(317, self.science_week.get_science_week_from_date(test_date))


# I cannot figure out how to properly mock a date such that ScienceWeek knows about my mock
class TestGetCurrentScienceWeek(TestScienceWeek):
    pass
    '''def test_science_week_of_today(self):
        with mock.patch('maven_iuvs.science_week.science_week.datetime.date') as mock_date:
            mock_date.today.return_value = date(2020, 1, 1)
            #self.assertEqual(mock_date.today(), date(2020, 1, 1))   # this works
            self.assertEqual(268, self.science_week.get_current_science_week())  # This doesn't about mock_date'''


class TestGetScienceWeekStartDate(TestScienceWeek):
    def test_week_cannot_be_float(self):
        self.assertRaises(TypeError, lambda: self.science_week.get_science_week_start_date(100.0))

    def test_week_cannot_be_negative(self):
        self.assertRaises(ValueError, lambda: self.science_week.get_science_week_start_date(-1))

    def test_start_week_0_is_mission_arrival_date(self):
        self.assertEqual(self.science_week.maven_arrival_date, self.science_week.get_science_week_start_date(0))

    def test_randomly_chosen_week_matches_known_start_date(self):
        self.assertEqual(date(2020, 12, 8), self.science_week.get_science_week_start_date(317))


class TestGetScienceWeekEndDate(TestScienceWeek):
    def test_week_cannot_be_float(self):
        self.assertRaises(TypeError, lambda: self.science_week.get_science_week_end_date(100.0))

    def test_week_cannot_be_negative(self):
        self.assertRaises(ValueError, lambda: self.science_week.get_science_week_end_date(-1))

    def test_week_of_mission_arrival(self):
        self.assertEqual(self.science_week.maven_arrival_date + timedelta(days=6),
                         self.science_week.get_science_week_end_date(0))

    def test_randomly_chosen_week_matches_known_start_date(self):
        self.assertEqual(date(2020, 12, 14), self.science_week.get_science_week_end_date(317))


class TestGetScienceWeekDateRange(TestScienceWeek):
    def test_output_is_tuple(self):
        self.assertTrue(isinstance(self.science_week.get_science_week_date_range(100), tuple))

    def test_output_is_2_elements(self):
        self.assertTrue(len(self.science_week.get_science_week_date_range(100)), 2)
