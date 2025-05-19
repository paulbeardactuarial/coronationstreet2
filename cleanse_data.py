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


# %%


all_dates = c_life_death.melt(
    id_vars="Character",
    value_vars=["Born", "Died", "First appearance", "Last appearance"],
    value_name="Value"
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
all_dates_clean = all_dates.loc[:, ["Field", "Character"]].merge(
    cleansed_dates_combined, left_index=True, right_index=True)
all_dates_clean = all_dates_clean.pivot(
    index="Character",
    values="Date",
    columns="Field"
).reset_index()


# %%

# re-construct the Corrie data set, using cleansed dates
df = corrie_data[corrie_data["Field"].isin(
    ["Character", "Occupation", "Duration"])].pivot(columns=["Field"], index="Character")
df = df.droplevel(level=0, axis=1)
df = df.reset_index()
df = df.merge(all_dates_clean, on=["Character"], how="outer")

# exit_classification_dict = {
#     "death": (df["Died"].notna() & df["Died"] <= df["Last appearance"]),
#     "alive": (df["Died"].isna() & df["Last appearance"].isna()),
#     "exit": (df["Died"].isna())
# }


(df["Died"] <= df["Last appearance"]).value_counts()
#
# exit_classification_dict
# exit_type = np.select()


# %%

# add a flag to determine if the character died whilst in exposure
# also create the exit date accordingly
