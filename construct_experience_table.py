# %%
import pandas as pd
import numpy as np
import datetime

# %%
input_fp = "./Data/character_data_segmented.parquet"
df = pd.read_parquet(input_fp)

# %%

# drop those without "Born" day
df = df[~df["Born"].isna()]


# %%

# def carve_experience_single_year(df):
exp_year = 2000


period_start = pd.Timestamp(datetime.date(exp_year, 1, 1))
period_end = pd.Timestamp(datetime.date(exp_year, 12, 31))

experience_start_field = "StartDate"
experience_end_field = "ExitDate"

# filter for rows that have experience within period
in_force_df = df[
    (df[experience_start_field] <= period_end) & (
        df[experience_end_field] >= period_start)
].copy()

for field in [experience_start_field, experience_end_field]:
    in_force_df[field] = in_force_df[field].clip(
        lower=period_start,
        upper=period_end
    )

# applying datetime.datetime.strptime()
# pd.dt.to_timestamp()
# pd.dt.to_period()


# .dt.to_timestamp()
# corrie_segments["Born"].year

# %%
