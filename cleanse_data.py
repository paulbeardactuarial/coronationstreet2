# %%
import pandas as pd
import numpy as np
import datetime
import re

# %%
corrie_data_fp = "./Data/character_data.csv"
corrie_data = pd.read_csv(corrie_data_fp)


# %%
c_life_death = corrie_data[corrie_data["Field"].isin(
    ["Born", "Died", "First appearance", "Last appearance", "Duration"])]
c_life_death = c_life_death.pivot(
    columns="Field", values="Value", index="Character").reset_index()
c_life_death

# %%
# triage the dates
date_string_series = c_life_death["Born"]

full_date_pattern = r'^\s*\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$'
month_year_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$'
year_pattern = r'(c.\s*|)\d{4}$'

date_string_formats = date_string_series.case_when(
    [
        (date_string_series.str.match(full_date_pattern).fillna(False), "full"),
        (date_string_series.str.match(month_year_pattern).fillna(False), "month_year"),
        (date_string_series.str.match(year_pattern).fillna(False), "year"),
        (date_string_series.notna(), "other")
    ]

)
date_string_formats.value_counts(dropna=False)
# date_string_series[date_string_series.str.match(year_pattern).fillna(False)]
# date_string_series[~date_string_series.str.match(full_date_pattern).fillna(True)]

# %%
date_strings = date_string_series.loc[
    date_string_series.str.match(full_date_pattern).fillna(False)
]
date_strings = date_strings.str.replace("st|nd|rd|th", "", n=1, regex=True)
date_strings = date_strings.apply(
    lambda x: datetime.datetime.strptime(x, "%d %B %Y")
)
date_strings
# end_year_regex = r'(\d{4})\s*$'
# mid_month_regex = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
# start_day_regex = r'^\s*(\d{1,2})'

# year = date_strings.str.extract(end_year_regex)[0]
# month = date_strings.str.extract(mid_month_regex)[0]
# day = date_strings.str.extract(start_day_regex)[0]


# %%
# define functions that will randomise the day or month of a date
def randomise_day_of_date(date_series, seed=42):

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


def randomise_month_of_date(date_series, seed=42):

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
date_strings_left = date_string_series.loc[
    ~date_string_series.str.match(full_date_pattern).fillna(True)
]


date_strings_year_month = date_strings_left[
    date_strings_left.str.match(month_year_only_pattern)
]

date_strings_year_month = date_strings_year_month.apply(
    lambda x: datetime.datetime.strptime(x, "%B %Y"))
