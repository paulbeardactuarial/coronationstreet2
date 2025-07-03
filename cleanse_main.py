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
corrie_data_fp = "./Data/character_data_enriched.csv"
corrie_master_df = pd.read_csv(corrie_data_fp)
corrie_master_df = corrie_master_df.pivot(
    columns="Field", values="Value", index="Character").reset_index()


# %%
# minor hacks to missing data...
index_frankie_b = corrie_master_df[corrie_master_df["Character"]
                                   == "Frankie Bolton"].index
corrie_master_df.loc[index_frankie_b, "Duration"] = '2024 to present'

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

# convert to datetime format
corrie_master_df.loc[:, date_fields_to_clean] = corrie_master_df.loc[:,
                                                                     date_fields_to_clean].apply(pd.to_datetime)


corrie_master_df = add_exit_info(
    corrie_master_df,
    present_date=pd.Timestamp(year=2025, month=6, day=30)
)

corrie_master_df["Number of appearances"] = pd.Series(
    corrie_master_df["Number of appearances"].str.replace(
        "\(.*\)", "", regex=True),
    dtype="int64")

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


# =============================================================================
# =============== create a multi-segment dataframe ============================
# =============================================================================

absence_df_2 = absence_df.reset_index().drop("index", axis=1)
absence_df_2 = absence_df_2.sort_values(["Character", "start_abscence"])
absence_df_rev = absence_df_2.sort_values(["start_abscence"], ascending=False)

absence_orderf = pd.Series(absence_df_2.groupby("Character")[
    "Character"].rank(method="first"), name="forward_order").astype(int)
absence_orderb = pd.Series(absence_df_rev.groupby("Character")[
    "Character"].rank(method="first",), name="backward_order").astype(int)
absence_df_2["orderf"] = absence_orderf
absence_df_2["orderb"] = absence_orderb

# calculate first block
first_block = absence_df_2[absence_df_2["orderf"] == 1]
first_block.loc[:, "exit_date"] = first_block.loc[:, "start_abscence"]
first_block = first_block.loc[:, ["Character", "start_date", "exit_date"]]

# calculate last block
last_block = absence_df_2[absence_df_2["orderb"] == 1]
last_block.loc[:, "start_date"] = last_block.loc[:, "end_abscence"]
last_block = last_block.loc[:, [
    "Character", "start_date", "exit_date"]]

# calculate the intermeidary blocks
other_blocks = absence_df_2
ob_start = other_blocks.loc[other_blocks.index[:-1], "end_abscence"]
ob_end = other_blocks.loc[other_blocks.index[1:], "start_abscence"]
ob_start.index = ob_start.index + 1
other_blocks["start_date"] = ob_start
other_blocks["exit_date"] = ob_end
other_blocks = other_blocks[(other_blocks["orderf"] != 1) & (
    other_blocks["start_date"] != other_blocks["exit_date"])]

other_blocks = other_blocks.loc[:, [
    "Character", "start_date", "exit_date"]]


# combine to get 'all_blocks' ...which contains the exposure breakdown
# (ONLY FOR THOSE WHO WERE ASBCENT)
all_blocks = pd.concat(
    [first_block, other_blocks, last_block]).sort_values(["Character", "start_date"])

all_blocks = all_blocks.reset_index().drop("index", axis=1)
all_blocks["Segment"] = pd.Series(all_blocks.groupby(
    "Character")["start_date"].rank().astype(int), name="Segment")

# multi
corrie_master_multis = (
    corrie_master_df
    .reindex(
        columns=np.append(np.setdiff1d(corrie_master_df.columns,
                                       all_blocks.columns), "Character"))
    .merge(all_blocks, on="Character", how="right")
    .loc[:, corrie_master_df.columns]
)

# multi segment no.
corrie_master_multis["Segment"] = pd.Series(
    corrie_master_multis.sort_values("start_date").groupby(
        "Character")["start_date"].rank().astype(int),
    name="Segment"
)

# max segment no.
corrie_master_multis["Max Segment"] = pd.Series(
    corrie_master_multis.groupby("Character")[
        "Segment"].transform(max).astype(int),
    name="Max Segment"
)

multi_chars = corrie_master_multis["Character"].unique()

# # filter out the singles only
corrie_master_singles = corrie_master_df[~corrie_master_df["Character"].isin(
    multi_chars)]

corrie_master_singles["Segment"] = 1
corrie_master_singles["Max Segment"] = 1

corrie_master_df_clean = pd.concat(
    [corrie_master_multis, corrie_master_singles])

corrie_master_df_clean = corrie_master_df_clean.sort_values(
    ["Character", "Segment"]).reset_index().drop("index", axis=1)


corrie_master_df_clean = corrie_master_df_clean.rename(
    lambda x: str.capitalize(x), axis=1)

corrie_master_df_clean.columns = corrie_master_df_clean.columns.str.replace(
    "_", " ")

corrie_master_df_clean["Exit status"] = corrie_master_df_clean["Exit status"].str.capitalize()

corrie_master_df_clean.loc[
    corrie_master_df_clean["Segment"] != corrie_master_df_clean["Max segment"],
    "Exit status"
] = "Exit"

corrie_master_df_clean = corrie_master_df_clean.set_index("Character")

# %%

corrie_master_df_clean.to_csv(
    "./Data/character_data_segmented.csv", index=True)

# %%
