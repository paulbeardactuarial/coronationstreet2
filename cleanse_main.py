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
# absence_df = absence_df.set_index("__temp_index__")
absence_summary = absence_df.assign(
    absence_length=absence_df["exit_date"] - absence_df["start_date"])

first_absence = pd.Series(
    absence_summary.groupby("Character")["start_date"].agg("min"),
    name="first_absence")

first_absence


# %%

absence_df = absence_df.sort_values(["Character", "start_abscence"])
absence_df_rev = absence_df.sort_values(["start_abscence"], ascending=False)

absence_orderf = pd.Series(absence_df.groupby("Character")[
    "Character"].rank(method="first"), name="forward_order").astype(int)
absence_orderb = pd.Series(absence_df_rev.groupby("Character")[
    "Character"].rank(method="first",), name="backward_order").astype(int)
absence_df["orderf"] = absence_orderf
absence_df["orderb"] = absence_orderb

# calculate first block
first_block = absence_df[absence_df["orderf"] == 1]
first_block.loc[:, "exit_date"] = first_block.loc[:, "start_abscence"]
first_block = first_block.loc[:, ["Character", "start_date", "exit_date"]]
first_block

# calculate last block
last_block = absence_df[absence_df["orderb"] == 1]
last_block.loc[:, "start_date"] = last_block.loc[:, "end_abscence"]
last_block = last_block.loc[:, [
    "Character", "start_date", "exit_date", "orderf"]]
last_block

# calculate other blocks
other_blocks = absence_df[(absence_df["orderf"] != 1)
                          & (absence_df["orderb"] != 1)]
# other_blocks["start_date"] = other_blocks["end_abscence"]
other_blocks


# %%
