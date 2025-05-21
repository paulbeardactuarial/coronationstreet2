# %%
import pandas as pd
import numpy as np
import datetime
import re
from itertools import chain

# %%
# function(s) to cleanse a single year string, which will be either:
# "yyyy"
# "yyyy-zzzz"
# "yyyytopresent"


def cleanse_duration(duration_str, present_year=2025):
    """
    Parses and expands a comma-separated string of years or year ranges into a sorted, unique list of years.

    Parameters:
    -----------
    duration_str : str
        A string containing years or year ranges separated by commas.
        Examples: "1999,2001-2003,2005-topresent"

    present_year : int, optional (default=2025)
        The value to use in place of 'topresent' in ranges.

    Returns:
    --------
    np.ndarray
        A sorted and unique array of integer years represented in the input string.
    """
    # convert to a list of integer years
    list_o_years = duration_str.split(",")
    list_o_years = [years for years in list_o_years if years]
    list_o_years = [cleanse_duration_component(
        i, present_year) for i in list_o_years]
    years_actual = np.array(list(chain.from_iterable(list_o_years)))
    years_actual = np.unique(years_actual)

    return years_actual


def cleanse_duration_component(single_duration, present_year=2025):
    """
    Converts a single year or a year range string into a list of individual years.

    Parameters:
    -----------
    single_duration : str
        A string representing a single year (e.g., "2005") or a range (e.g., "2001-2003", "2010-topresent").

    present_year : int, optional (default=2025)
        The value to substitute for 'topresent' in the range.

    Returns:
    --------
    list of int
        A list of individual years represented by the input.
    """
    single_duration = single_duration.replace(" ", "")
    single_duration = single_duration.replace("topresent", f"-{present_year}")

    assert (single_duration.count("-") <= 1)

    if (single_duration.count("-") == 1):
        single_duration_as_list = list(
            range(
                int(re.findall("^(\d{4})-\d{4}$", single_duration)[0]),
                int(re.findall("^\d{4}-(\d{4})$", single_duration)[0]) + 1
            )
        )
    else:
        single_duration_as_list = [int(single_duration)]

    return single_duration_as_list


def calculate_years_out_of_force(years_in_force_single: list[int]) -> list[int]:
    """
    Identifies the years that are *not* in force within the range defined by the minimum and maximum
    years present in the input list.

    Parameters:
    -----------
    years_in_force_single : list of int
        A list of individual years (e.g., [2001, 2002, 2004]) assumed to represent the years something
        was in force.

    Returns:
    --------
    list of int
        A list of years within the range from min(years_in_force_single) to max(years_in_force_single)
        that are *not* present in the input list.
    """
    years_continuous = np.arange(stop=years_in_force_single.max() + 1,
                                 start=years_in_force_single.min(), step=1)
    years_abscent = np.setdiff1d(years_continuous, years_in_force_single)
    return years_abscent


def extract_years_out_dict(years_abscent):
    """
    Converts a list or NumPy array of non-consecutive missing years into a dictionary of start and end years
    for each gap period.

    This is useful for identifying continuous blocks of years that were out of force.

    Parameters:
    -----------
    years_abscent : array-like (e.g., NumPy array)
        A sorted array of years that are not in force (i.e., missing from a range). Should be a NumPy array
        for compatibility with vectorized operations.

    Returns:
    --------
    dict
        A dictionary with two keys:
        - "start_abscence": List of start years just before each missing interval.
        - "end_abscence": List of end years just after each missing interval.

        Example:
            Input: [2003, 2004, 2007]
            Output: {"start_abscence": [2002, 2006], "end_abscence": [2005, 2008]}
    """
    if len(years_abscent) == 0:
        return {"start_abscence": None, "end_abscence": None}
    if len(years_abscent) == 1:
        output = {
            "start_abscence": (years_abscent - 1).astype(int).tolist(),
            "end_abscence": (years_abscent + 1).astype(int).tolist()
        }
    else:
        start_abscent = years_abscent - 1
        end_abscent = years_abscent
        unique_start_abscent = start_abscent[1:] != end_abscent[:-1]

        output = {
            "start_abscence": (start_abscent[np.append(True, unique_start_abscent)]).astype(int).tolist(),
            "end_abscence": (end_abscent[np.append(unique_start_abscent, True)] + 1).astype(int).tolist()
        }
    return output


def construct_absence_df(years_out_force):
    years_out_summ = years_out_force.apply(lambda x: extract_years_out_dict(x))
    years_out_start = years_out_summ.apply(lambda x: x["start_abscence"])
    years_out_end = years_out_summ.apply(lambda x: x["end_abscence"])

    absences = pd.DataFrame(
        {
            "start_abscence": years_out_start,
            "end_abscence": years_out_end
        },
        index=years_out_summ.index
    ).explode(["start_abscence", "end_abscence"]).dropna()

    absences = absences.apply(
        lambda s: pd.to_datetime((s.astype(str) + "-06-30")),
        axis=1
    )
    return absences


def process_absence_data(absences, df):
    """
    Merges absence data with character metadata and filters/adjusts date ranges.

    This function performs the following steps:
        1. Merges the `absences` DataFrame with the `df` DataFrame on the index.
        2. Filters out absence periods that occur completely outside the character's
           active range, defined by 'start_date' and 'exit_date'.
        3. Adjusts ('floors' and 'ceilings') the absence period to fit strictly within
           the character's valid range, adding a 1-day cushion to avoid zero-day overlaps.
        4. Returns a cleaned DataFrame with selected columns and renamed date fields.

    Parameters:
        absences (pd.DataFrame): A DataFrame containing absence records with at least:
            - 'start_abscence' (datetime): The beginning of the absence period.
            - 'end_abscence' (datetime): The end of the absence period.
        df (pd.DataFrame): A DataFrame with character information, including:
            - 'start_date' (datetime): The start of character activity.
            - 'exit_date' (datetime): The end of character activity.
            - 'Character' (str), 'exit_status' (str), 'First appearance' (datetime).

    Returns:
        pd.DataFrame: A cleaned and aligned DataFrame of absences with the following columns:
            - 'Character'
            - 'start_date' (adjusted from 'start_abscence')
            - 'exit_date' (adjusted from 'end_abscence')
            - 'exit_status'
            - 'First appearance'
    """
    # Merge absences with character data
    adf = absences.merge(df, how="left", left_index=True, right_index=True)
    adf = adf.reset_index(names="__temp_index__")

    # Filter out absences that are completely outside the character's valid time range
    absence_before_start = adf["end_abscence"] < adf["start_date"]
    absence_after_exit = adf["start_abscence"] > adf["exit_date"]
    adf = adf[~(absence_before_start | absence_after_exit)]

    # Adjust (floor/ceiling) absence periods to ensure they fit within valid bounds
    floored_start = np.maximum(
        adf["start_abscence"], adf["start_date"] + pd.Timedelta(1, "day"))
    ceilinged_end = np.minimum(
        adf["end_abscence"], adf["exit_date"] - pd.Timedelta(1, "day"))

    adf["start_abscence"] = floored_start
    adf["end_abscence"] = ceilinged_end

    # Select and rename relevant columns
    absence_df = adf[["__temp_index__",
                      "Character",
                      "start_abscence",
                      "end_abscence",
                      "exit_status",
                      "First appearance"]]
    absence_df = absence_df.rename(columns={
        "start_abscence": "start_date",
        "end_abscence": "exit_date",
    })

    absence_df.set_index("__temp_index__")

    return absence_df


# %%

# adf = absences.merge(df, how="left", left_index=True, right_index=True)

# # filter out absences that are beyond start_date --> exit_date range
# absence_before_start = adf["end_abscence"] < adf["start_date"]
# absence_after_exit = adf["start_abscence"] > adf["exit_date"]
# adf = adf[~(absence_before_start | absence_after_exit)]

# # floor/ceiling (add 1 day cushion just to ensure no chance of zero exposure for a block with a death)
# floored_start = np.maximum(
#     adf["start_abscence"], adf["start_date"] + pd.Timedelta(1, "day"))
# ceilinged_end = np.minimum(
#     adf["end_abscence"], adf["exit_date"] - pd.Timedelta(1, "day"))

# adf["start_abscence"] = floored_start
# adf["end_abscence"] = ceilinged_end

# absence_df = adf[["Character", "start_abscence",
#                   "end_abscence", "exit_status", "First appearance"]]
# absence_df = absence_df.rename(columns={
#     "start_abscence": "start_date",
#     "end_abscence": "exit_date",
# })
# absence_df


# %%
# remove text in brackets from 'Duration'


# %%
# abscent_gaps = pd.Series(
#     years_out_start.dropna().apply(len), name='Abscent Gaps')

# abscent_years = pd.Series(
#     years_out_force.dropna().apply(len), name='Abscent Years')

# abscent_years.argmax()
