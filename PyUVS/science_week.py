# Built-in imports
from datetime import date, timedelta


class ScienceWeek:
    """A ScienceWeek object can convert dates into MAVEN science weeks"""
    def __init__(self):
        """
        Attributes
        ----------
        maven_arrival_date: datetime.date
            The date when MAVEN arrived at Mars
        """
        self.maven_arrival_date = date(2014, 11, 11)

    def get_science_week_from_date(self, some_date):
        """Get the science week number at an input date

        Parameters
        ----------
        some_date: datetime.date
            The date at which to get the science week

        Returns
        -------
        int of the science week
        """
        self.__check_input_is_datetime_date(some_date, 'some_date')
        self.__check_date_not_before_mission_arrival(some_date)
        return (some_date - self.maven_arrival_date).days // 7

    def get_current_science_week(self):
        """Get the science week number for today

        Returns
        -------
        int of the science week
        """
        return self.get_science_week_from_date(date.today())

    def get_science_week_start_date(self, week):
        """Get the date when a science week began

        Parameters
        ----------
        week: int
            The science week

        Returns
        -------
        datetime.date of day the science week started
        """
        self.__check_science_week_is_int(week)
        self.__check_science_week_is_nonnegative(week)
        return self.maven_arrival_date + timedelta(days=week * 7)

    def get_science_week_end_date(self, week):
        """Get the date when a science week ended

        Parameters
        ----------
        week: int
            The science week

        Returns
        -------
        datetime.date of day the science week ended
        """
        self.__check_science_week_is_int(week)
        self.__check_science_week_is_nonnegative(week)
        return self.get_science_week_start_date(week + 1) - timedelta(days=1)

    def get_science_week_date_range(self, week):
        """Get the date range corresponding to the input science week

        Parameters
        ----------
        week: int
            The science week

        Returns
        -------
        The science week start and end dates
        """
        return self.get_science_week_start_date(week), self.get_science_week_end_date(week)

    @staticmethod
    def __check_input_is_datetime_date(some_date, date_name):
        if not isinstance(some_date, date):
            raise TypeError(f'{date_name} must be a datetime.date.')

    @staticmethod
    def __check_science_week_is_int(week):
        if not isinstance(week, int):
            raise TypeError('week must be an int.')

    def __check_date_not_before_mission_arrival(self, some_date):
        time_delta = (some_date - self.maven_arrival_date).days
        if time_delta < 0:
            raise ValueError('The input date is before MAVEN arrived at Mars.')

    @staticmethod
    def __check_science_week_is_nonnegative(week):
        if week < 0:
            raise ValueError('week cannot be a negative value')
