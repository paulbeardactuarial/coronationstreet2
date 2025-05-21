# %%
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import datetime
    import re
    from itertools import chain

import cleanse_data
import cleanse_absences


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
# re-construct the Corrie data set, using cleansed dates
df = corrie_data[corrie_data["Field"].isin(
    ["Character", "Occupation", "Duration"])].pivot(columns=["Field"], index="Character")
df = df.droplevel(level=0, axis=1)
df = df.reset_index()
df = df.merge(all_dates_clean, on=["Character"], how="outer")


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

present_year = 2025
df = pd.DataFrame()
duration_series = df["Duration"]

duration_series = duration_series.str.replace("\(.*\)", ",", regex=True)

years_in_force = duration_series.dropna().apply(
    lambda x: cleanse_duration(x, present_year))

years_out_force = years_in_force.apply(
    lambda x: calculate_years_out_of_force(x))

absences = construct_absence_df(years_out_force)
absence_df = process_absence_data(absences, df)
