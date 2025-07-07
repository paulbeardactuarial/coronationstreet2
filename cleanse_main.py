# %%

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import datetime
    import re
    from itertools import chain

from cleanse_data import *
from cleanse_absences import *
from cleaner_functions import clean_column_names, years_diff

# %%

corrie_data_fp = "./Data/character_data_enriched.csv"
corrie_master_df = pd.read_csv(corrie_data_fp)
corrie_master_df = corrie_master_df.pivot(
    columns="Field", values="Value", index="Character"
).reset_index()

# %%

index_frankie_b = corrie_master_df[corrie_master_df["Character"]
                                   == "Frankie Bolton"].index
corrie_master_df.loc[index_frankie_b, "Duration"] = '2024 to present'

# %%

date_fields_to_clean = ["Born", "Died", "First appearance", "Last appearance"]
all_dates_clean = extract_and_clean_dates(
    corrie_master_df, date_fields_to_clean)

corrie_master_df = corrie_master_df.reindex(
    columns=corrie_master_df.columns.difference(all_dates_clean.columns)
).merge(
    all_dates_clean, left_index=True, right_index=True, how="left"
)

corrie_master_df[date_fields_to_clean] = corrie_master_df[date_fields_to_clean].apply(
    pd.to_datetime)

corrie_master_df = add_exit_info(
    corrie_master_df,
    present_date=pd.Timestamp(year=2025, month=6, day=30)
)

corrie_master_df["Number of appearances"] = pd.Series(
    corrie_master_df["Number of appearances"].str.replace(
        r"\(.*\)", "", regex=True),
    dtype="int64"
)

# %%

present_year = 2025
duration_series = corrie_master_df["Duration"].str.replace(
    r"\(.*\)", ",", regex=True)

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

absence_df_2["orderf"] = absence_df_2.groupby(
    "Character")["Character"].rank(method="first").astype(int)
absence_df_2["orderb"] = absence_df_rev.groupby(
    "Character")["Character"].rank(method="first").astype(int)

first_block = absence_df_2[absence_df_2["orderf"] == 1]
first_block.loc[:, "exit_date"] = first_block["start_abscence"]
first_block = first_block[["Character", "start_date", "exit_date"]]

last_block = absence_df_2[absence_df_2["orderb"] == 1]
last_block.loc[:, "start_date"] = last_block["end_abscence"]
last_block = last_block[["Character", "start_date", "exit_date"]]

other_blocks = absence_df_2.copy()
ob_start = other_blocks.loc[other_blocks.index[:-1], "end_abscence"]
ob_end = other_blocks.loc[other_blocks.index[1:], "start_abscence"]
ob_start.index = ob_start.index + 1
other_blocks["start_date"] = ob_start
other_blocks["exit_date"] = ob_end

other_blocks = other_blocks[
    (other_blocks["orderf"] != 1) &
    (other_blocks["start_date"] != other_blocks["exit_date"])
]
other_blocks = other_blocks[["Character", "start_date", "exit_date"]]

all_blocks = pd.concat([first_block, other_blocks, last_block]).sort_values(
    ["Character", "start_date"]
).reset_index(drop=True)

all_blocks["Segment"] = all_blocks.groupby(
    "Character")["start_date"].rank().astype(int)

corrie_master_multis = (
    corrie_master_df
    .reindex(columns=np.append(np.setdiff1d(corrie_master_df.columns, all_blocks.columns), "Character"))
    .merge(all_blocks, on="Character", how="right")
    .loc[:, corrie_master_df.columns]
)

corrie_master_multis["Segment"] = corrie_master_multis.sort_values("start_date").groupby(
    "Character"
)["start_date"].rank().astype(int)

corrie_master_multis["Max Segment"] = corrie_master_multis.groupby(
    "Character")["Segment"].transform(max).astype(int)

multi_chars = corrie_master_multis["Character"].unique()
corrie_master_singles = corrie_master_df[~corrie_master_df["Character"].isin(
    multi_chars)]
corrie_master_singles["Segment"] = 1
corrie_master_singles["Max Segment"] = 1

corrie_master_df_clean = pd.concat([corrie_master_multis, corrie_master_singles]).sort_values(
    ["Character", "Segment"]
).reset_index(drop=True)

corrie_master_df_clean = corrie_master_df_clean.rename(str.capitalize, axis=1)
corrie_master_df_clean.columns = corrie_master_df_clean.columns.str.replace(
    "_", " ")
corrie_master_df_clean["Exit status"] = corrie_master_df_clean["Exit status"].str.capitalize()

corrie_master_df_clean.loc[
    corrie_master_df_clean["Segment"] != corrie_master_df_clean["Max segment"],
    "Exit status"
] = "Exit"

corrie_master_df_clean = corrie_master_df_clean.set_index("Character")

# %%

# =============================================================================
# =============== apply some more column cleaning ============================
# =============================================================================


corrie_master_df_clean.columns = clean_column_names(
    corrie_master_df_clean.columns)
date_cols = ['Born', 'Died', 'ExitDate',
             'StartDate', 'FirstAppearance', 'LastAppearance']

corrie_master_df_clean.loc[:, date_cols] = corrie_master_df_clean.apply(
    {col: lambda x: pd.to_datetime(x) for col in date_cols}
)
corrie_master_df_clean.loc[
    corrie_master_df_clean["ExitStatus"] != "Death",
    "Died"
] = pd.NA

corrie_master_df_clean = corrie_master_df_clean.rename(
    {"Sic": "Industry"}, axis=1)
corrie_master_df_clean["Returner"] = corrie_master_df_clean["MaxSegment"] > 1


# %%

# =============================================================================
# =============== select and save dataframe ============================
# =============================================================================

column_metadata = {
    "Gender": "Character gender (M/F)",
    "NumberOfAppearances": "Total number of appearances",
    "FirstAppearance": "Datetime of first appearance",
    "LastAppearance": "Datetime of last appearance",
    "ExitStatus": "Character's exit status (Alive/Death/Exit)",
    "Born": "Datetime of character's birth (if known)",
    "Died": "Datetime of character's death (if occcurred)",
    "Returner": "Boolean, True if character returned after exit",
    "NumberTimesMarried": "Number of times character married",
    "NumberChildren": "Number of children character has",
    "Occupation": "Character's occupation",
    "Industry": "Industry sector of occupation",
    "BigamyCommitted": "Whether character was involved in bigamous marriage (boolean)",
    "StartDate": "Datetime when the character started this segment on the show",
    "ExitDate": "Datetime when the character exited the show for this segment",
    "Segment": "Sequential number for each continuous appearance block for the character",
    "MaxSegment": "Total number of appearance segments for the character"
}

corrie_master_df_clean.attrs["column_metadata"] = column_metadata
corrie_master_df_clean.attrs["data_date"] = "2025-06-30"
corrie_master_df_clean = corrie_master_df_clean[list(column_metadata.keys())]

corrie_master_df_clean.to_parquet(
    "./Data/character_data_segmented.parquet", engine="pyarrow")

# %%
