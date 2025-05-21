# %%
import pandas as pd
import numpy as np
import datetime
import re
from itertools import chain


# %%
# define function to triage the dates


def triage_date_string_fmt(
        date_string_series,
        string_patterns={
            "full": r'^\s*\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$',
            "month_year": r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$',
            "year": r'(c.\s*|)\d{4}$',
            "multi": r'(\s*\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*)',
            "other": r'.'
        }
):

    patterns = list(string_patterns.values())
    format_names = list(string_patterns.keys())

    conditions = [date_string_series.str.match(
        pattern).fillna(False) for pattern in patterns]

    date_string_formats = pd.Series(
        np.select(
            condlist=conditions,
            choicelist=format_names,
            default=pd.NA
        ),
        index=date_string_series.index
    )

    return date_string_formats


# %%
# define functions that will randomise the day or month of a date
def cleanse_month_year(date_series, seed=42):

    date_series = date_series.apply(
        lambda x: datetime.datetime.strptime(x, "%B %Y")
    )

    # reset date series to first of month (if isn't already)
    date_series = date_series.dt.to_period('M').dt.to_timestamp()

    # random seed generator
    rndm = np.random.default_rng(seed)

    # create random integers of days to add
    min_day = np.ones(len(date_series))
    max_day = date_series.dt.daysinmonth
    random_day = rndm.integers(min_day - 1, max_day, len(date_series))
    random_day = pd.to_timedelta(random_day, unit="days")

    # add days to date_series
    rndmised_date_series = date_series + random_day

    return rndmised_date_series


def cleanse_year(date_series, seed=42):

    date_series = date_series.str.replace("c.", "").str.strip()
    date_series = date_series.apply(
        lambda x: datetime.datetime.strptime(x, "%Y")
    )

    # reset date series to first of month (if isn't already)
    date_series = date_series.dt.to_period('Y').dt.to_timestamp()

    # random seed generator
    rndm = np.random.default_rng(seed)

    # create random integers of days to add
    min_month = 1
    max_month = 12
    random_month = rndm.integers(min_month - 1, max_month, len(date_series))
    random_month = pd.Series(
        [pd.DateOffset(months=month) for month in random_month],
        index=date_series.index
    )

    # add days to date_series
    rndmised_date_series = date_series + random_month

    return rndmised_date_series


# %%
# functions that handle cleansing process

def cleanse_full(date_string_series):
    date_string_series = date_string_series.str.replace(
        "st|nd|rd|th", "", n=1, regex=True)
    date_strings = date_string_series.apply(
        lambda x: datetime.datetime.strptime(x, "%d %B %Y"))
    return date_strings


def cleanse_full_incl_multi(
        date_string_series,
        full_date_pattern=r'(\s*\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*)'
):
    multi_full_extract = date_string_series.str.extractall(full_date_pattern)[
        0]
    first_full_extract = multi_full_extract.xs(key=0, level=1).str.strip()
    cleansed_date_string_series = first_full_extract.combine_first(
        date_string_series)
    date_strings = cleanse_full(cleansed_date_string_series)

    return date_strings


def extract_and_clean_dates(
    df: pd.DataFrame,
    date_fields: list = ["Born", "Died", "First appearance", "Last appearance"]
) -> pd.DataFrame:
    """
    Extracts and cleans date-related fields from a character-based dataframe.
    The function will call the following cleansing functions...
    - triage_date_string_fmt
    - cleanse_full
    - cleanse_month_year
    - cleanse_year  
    - cleanse_full_incl_multi

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing character information and various date-related columns.

    date_fields : list
        A list of column names in `df` that contain date information (e.g., ["Born", "Died", "First appearance", "Last appearance"]).

    Returns:
    -------
    pd.DataFrame
        A DataFrame with one row per character and cleaned date columns, reshaped into wide format.
    """

    all_dates = df.melt(
        value_vars=date_fields,
        value_name="Value",
        ignore_index=False
    )

    date_string_series = all_dates["Value"].str.strip()
    date_string_formats = triage_date_string_fmt(date_string_series)

    # get the strings for each type
    full_date_strings = date_string_series[date_string_formats == "full"]
    month_year_date_strings = date_string_series[date_string_formats == "month_year"]
    year_strings = date_string_series[date_string_formats == "year"]
    multi_strings = date_string_series[date_string_formats == "multi"]

    # get the cleansed strings for each type
    cleansed_dates = {
        "full": cleanse_full(full_date_strings),
        "month_year": cleanse_month_year(month_year_date_strings),
        "year": cleanse_year(year_strings),
        "multi": cleanse_full_incl_multi(multi_strings)
    }

    # get an NA array for anything not triggered
    na_date_string_series = pd.Series(np.nan, index=date_string_series.index)

    # stitch it all back together
    cleansed_dates_combined = pd.Series(na_date_string_series, name="Date")
    for key in cleansed_dates.keys():
        cleansed_dates_combined = cleansed_dates_combined.combine_first(
            cleansed_dates[key])

    # ...and pivot back into wider df
    all_dates_clean = all_dates.loc[:, ["Field"]].merge(
        cleansed_dates_combined, left_index=True, right_index=True)

    all_dates_clean = all_dates_clean.pivot(
        values="Date",
        columns="Field"
    )

    return all_dates_clean


# %%


def add_exit_info(df):
    """
    Adds exit information to a DataFrame based on character status.

    This function classifies each character's exit status and calculates their corresponding
    exit date based on the values in the 'Died', 'Last appearance', and 'Duration' columns.

    The logic for classifying `exit_status` is as follows:
        - "death": The character died within 12 months of their 'Last appearance'.
        - "alive": The character has not died (NaN in 'Died') and their 'Duration' contains "present".
        - "exit": The character has a non-null 'Last appearance' but doesn't meet the criteria for "death" or "alive".

    The function also computes the corresponding `exit_date`:
        - For "death": The date of death.
        - For "alive": The current date.
        - For "exit": The 'Last appearance' date.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following required columns:
            - 'Died' (datetime or NaT): Date the character died.
            - 'Last appearance' (datetime or NaT): Date of the character's last appearance.
            - 'Duration' (str): String indicating the time range the character appeared, e.g., "2005–present".

    Returns:
        pd.DataFrame: The original DataFrame with two additional columns:
            - 'exit_status' (str): One of "death", "alive", or "exit".
            - 'exit_date' (datetime): The corresponding exit date based on classification.
    """
    exit_classification_dict = {
        # we are classifying as `death` is died within 12 months of `Last appearance`
        "death": df["Died"] <= df["Last appearance"] + pd.DateOffset(years=1),
        "alive": df["Died"].isna() & df["Duration"].str.contains("present", case=False),
        "exit": df["Last appearance"].notna()
    }

    exit_status = pd.Series(
        np.select(
            condlist=list(exit_classification_dict.values()),
            choicelist=list(exit_classification_dict.keys()),
            default=None
        ),
        name="exit_status"
    )

    present_date = pd.Series(pd.Timestamp.now().floor('D'),
                             index=df.index, dtype='<M8[ns]')

    exit_date = pd.Series(
        np.select(
            condlist=list(exit_classification_dict.values()),
            choicelist=[
                df["Died"],  # death
                present_date,  # alive
                df["Last appearance"]  # exit
            ],
            default=None
        ),
        name="exit_date",
        dtype='<M8[ns]'
    )

    df["exit_status"] = exit_status
    df["exit_date"] = exit_date

    return df
