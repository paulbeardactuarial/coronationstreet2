# %%
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import datetime
    import re
    from itertools import chain

from cleanse_data import *
from cleanse_absences import *


# %%
corrie_data_fp = "./Data/character_data.csv"
corrie_master_df = pd.read_csv(corrie_data_fp)
corrie_master_df = corrie_master_df.pivot(
    columns="Field", values="Value", index="Character").reset_index()


# %%

# replace date fields with the cleansed date fields from 'all_dates_clean'

date_fields_to_clean = ["Born", "Died", "First appearance", "Last appearance"]

# get cleansed date fields
all_dates_clean = extract_and_clean_dates(
    corrie_master_df,
    date_fields_to_clean
)

# strip the columns from 'corrie_master_df' that are in 'all_dates_clean'
corrie_master_df = corrie_master_df.reindex(
    columns=corrie_master_df.columns.difference(all_dates_clean.columns)
)

corrie_master_df = corrie_master_df.merge(
    all_dates_clean,
    left_index=True,
    right_index=True,
    how="left"
)

corrie_master_df = add_exit_info(corrie_master_df)


# %%

present_year = 2025
duration_series = corrie_master_df["Duration"]

duration_series = duration_series.str.replace("\(.*\)", ",", regex=True)

years_in_force = duration_series.dropna().apply(
    lambda x: cleanse_duration(x, present_year))

years_out_force = years_in_force.apply(
    lambda x: calculate_years_out_of_force(x))

absences = construct_absence_df(years_out_force)

absence_df = process_absence_data(absences, corrie_master_df)

# %%
