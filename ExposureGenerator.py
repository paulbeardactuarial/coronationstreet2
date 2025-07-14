import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Union


class ExposureGenerator():
    """
    ExposureGenerator generates exposure periods from a DataFrame containing individual records
    (e.g., insurance policies or life events) with start and end dates. It provides methods to
    split exposure periods by anniversary dates, clip periods to experience boundaries, and assign
    exposures and event indicators for each period.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing individual records.
        date_fields (List[str]): List of date field names used for carving exposure periods (e.g., birthdate).
        experience_start_field (str): Name of the field indicating the start of experience (default: "StartDate").
        experience_end_field (str): Name of the field indicating the end of experience (default: "ExitDate").

    Methods:
        set_df(df):
            Set a new DataFrame for the generator.
        set_date_fields(date_fields):
            Set new date fields for carving exposure periods.
        construct_exposure_single_period(period_start):
            Construct the exposure DataFrame for a single period starting at period_start.
        construct_exposure_full(month=1, day=1):
            Construct the full exposure DataFrame across all periods, starting at the specified month and day each year.

    Internal Methods:
        _extract_anniversary(period_df, date_field, period_start):
            Calculate the anniversary date for each record for the given date_field and period_start.
            Handles leap years by assigning Feb 28 for Feb 29 anniversaries in non-leap years.
        _extract_years_at_start(period_df, date_field, period_start):
            Calculate the integer number of years since the date_field for each record at the period_start.
        _initialise_period_df(period_start):
            Create a DataFrame of records in force at the start of the period, with period start and end dates.
        _carve_by_date(period_df, date_field, period_start):
            Split period_df into sub-periods at the anniversary of date_field, assigning years since the date.
        _carve_by_dates(period_df, period_start):
            Sequentially carve period_df by all date_fields.
        _clip_period_dates(period_df):
            Clip the period start and end dates to the experience start and end dates for each record.
        _assign_exposure(period_df):
            Assign fractional year exposure to each period based on the number of days in the period.
        _assign_death(period_df):
            Assign a boolean indicator for whether a death event occurs within the period.

    Usage:
        Instantiate ExposureGenerator with a DataFrame and relevant date fields, then use
        construct_exposure_full() to generate a DataFrame of exposures split by period and carved by anniversaries.
    """

    # Class constants
    DAYS_PER_YEAR = 365.25

    def __init__(self,
                 df: pd.DataFrame,
                 date_fields: List[str],
                 experience_start_field: str = "StartDate",
                 experience_end_field: str = "ExitDate",
                 death_status_field: str = "ExitStatus") -> None:

        self.df = df
        self.date_fields = date_fields
        self.experience_start_field = experience_start_field
        self.experience_end_field = experience_end_field
        self.death_status_field = death_status_field

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas Dataframe")

    def set_df(self, df):
        self.df = df

    def set_date_fields(self, date_fields):
        self.date_fields = date_fields

    def _extract_anniversary(self, period_df, date_field, period_start):
        month = period_df[date_field].dt.month
        day = period_df[date_field].dt.day
        is_feb_29 = (month == 2) & (day == 29)
        anniv_year = pd.Series(period_start.year, index=period_df.index)
        anniversary = pd.to_datetime({
            'year': anniv_year,
            'month': month,
            'day': day
        }, errors='coerce')
        anniversary[anniversary.isna() & is_feb_29] = pd.to_datetime(
            {
                'year': anniv_year[anniversary.isna()],
                'month': 2,
                'day': 28
            }
        )
        anniversary = anniversary.where(
            anniversary >= period_start,
            anniversary + pd.DateOffset(years=1)
        )
        anniversary.name = f"Anniv{date_field}"
        return anniversary

    def _extract_years_at_start(self, period_df, date_field, period_start):
        integer_years_at_start = pd.Series(
            period_df[date_field].apply(
                lambda d: relativedelta(period_start, d).years),
            name=f"YearsSince{date_field}",
            index=period_df.index
        )
        return integer_years_at_start

    def _initialise_period_df(self, period_start):
        df = self.df
        period_end = pd.Timestamp(datetime.date(
            period_start.year+1,
            period_start.month,
            period_start.day)
        )
        in_force_df = df[
            (df[self.experience_start_field] <= period_end) & (
                df[self.experience_end_field] >= period_start)
        ].copy()
        in_force_df["PeriodStartDate"] = period_start
        in_force_df["PeriodEndDate"] = period_end
        return in_force_df

    def _carve_by_date(self, period_df, date_field, period_start):
        new_field_name = f"YearsSince{date_field}"
        # get the anniversary and years at start
        ann = self._extract_anniversary(period_df, date_field, period_start)
        yea = self._extract_years_at_start(period_df, date_field, period_start)

        # split the dataframe into 3

        rows_to_cut = (
            period_df["PeriodStartDate"] < ann) & (
            period_df["PeriodEndDate"] > ann
        )
        uncut_df = period_df[~rows_to_cut].copy()
        pre_df = period_df[rows_to_cut].copy()
        post_df = pre_df.copy()

        uncut_df[new_field_name] = yea[~rows_to_cut]

        pre_df["PeriodEndDate"] = ann[rows_to_cut]
        pre_df[new_field_name] = yea[rows_to_cut]

        post_df["PeriodStartDate"] = ann[rows_to_cut]
        post_df[new_field_name] = yea[rows_to_cut] + 1

        output = pd.concat([uncut_df, pre_df, post_df])
        new_index = [*period_df.index.names, new_field_name]
        output = output.reset_index().set_index(new_index)
        return output

    def _carve_by_dates(self, period_df, period_start):
        for date_field in self.date_fields:
            period_df = self._carve_by_date(
                period_df, date_field, period_start)
        return period_df

    def _clip_period_dates(self, period_df):
        for field in ["PeriodStartDate", "PeriodEndDate"]:
            period_df[field] = period_df[field].clip(
                lower=period_df[self.experience_start_field],
                upper=period_df[self.experience_end_field] +
                pd.Timedelta(days=1)
            )
        return period_df

    def _assign_exposure(self, period_df):
        period_df["Exposure"] = (
            (period_df["PeriodEndDate"] -
             period_df["PeriodStartDate"])
            .dt
            .days
            .clip(lower=0))/self.DAYS_PER_YEAR
        return period_df

    def _assign_death(self, period_df):
        period_df["DeathCount"] = (
            (period_df[self.death_status_field] == "Death") &
            (period_df[self.experience_end_field] < period_df["PeriodEndDate"]) &
            (period_df[self.experience_end_field]
             >= period_df["PeriodStartDate"])
        ).astype(int)
        return period_df

    def construct_exposure_single_period(self, period_start: pd.Timestamp) -> pd.DataFrame:
        """
        Construct the exposure DataFrame for a single period starting at period_start.

        Args:
            period_start: The start date of the exposure period (as a pandas Timestamp).

        Returns:
            DataFrame with exposure periods for records in force at period_start,
            carved by anniversaries, clipped to experience boundaries, and with
            exposure and death indicators assigned. Returns None if no records are in force.
        """
        period_df = self._initialise_period_df(period_start)
        if len(period_df) == 0:
            return None
        period_df = self._carve_by_dates(period_df, period_start)
        period_df = self._clip_period_dates(period_df)
        period_df = self._assign_exposure(period_df)
        period_df = self._assign_death(period_df)
        return period_df

    def construct_exposure_full(self, month: int = 1, day: int = 1) -> pd.DataFrame:
        """
        Construct exposure DataFrame across all periods.

        Args:
            month: Starting month for each period (default: 1)
            day: Starting day for each period (default: 1)

        Returns:
            DataFrame with exposure periods carved by anniversaries
        """
        df = self.df
        years_to_analyse = range(
            df[self.experience_start_field].min().year - 1,
            df[self.experience_end_field].max().year + 1,
            1)
        period_starts = [
            pd.Timestamp(datetime.date(i, month, day)) for i in years_to_analyse]
        exposure_full = [self.construct_exposure_single_period(
            period_start) for period_start in period_starts]
        exposure_full = [df for df in exposure_full if df is not None]
        if not exposure_full:
            return pd.DataFrame()

        return pd.concat(exposure_full)
