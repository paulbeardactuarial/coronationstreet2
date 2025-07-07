# %%
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from cleaner_functions import years_diff

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
period_end = pd.Timestamp(datetime.date(
    period_start.year+1,
    period_start.month,
    period_start.day)
)

experience_start_field = "StartDate"
experience_end_field = "ExitDate"
date_fields = ["Born", "FirstAppearance"]
date_field = "Born"

# filter for rows that have experience within period
in_force_df = df[
    (df[experience_start_field] <= period_end) & (
        df[experience_end_field] >= period_start)
].copy()

# for field in [experience_start_field, experience_end_field]:
#     in_force_df[field] = in_force_df[field].clip(
#         lower=period_start,
#         upper=period_end
#     )


def extract_anniversary(df, date_field, period_start):
    # Extract month and day
    month = df[date_field].dt.month
    day = df[date_field].dt.day

    # Handle Feb 29 in non-leap years: fallback to Feb 28
    is_feb_29 = (month == 2) & (day == 29)

    # Create anniversary datetime safely
    anniv_year = pd.Series(period_start.year, index=df.index)

    # Start with default values
    anniversary = pd.to_datetime({
        'year': anniv_year,
        'month': month,
        'day': day
    }, errors='coerce')

    # For failed conversions (e.g. Feb 29 on non-leap year), fallback to Feb 28
    anniversary[anniversary.isna() & is_feb_29] = pd.to_datetime(
        {
            'year': anniv_year[anniversary.isna()],
            'month': 2,
            'day': 28
        }
    )

    # Adjust to next year if before period_start
    anniversary = anniversary.where(
        anniversary >= period_start,
        anniversary + pd.DateOffset(years=1)
    )

    anniversary.name = f"Anniv{date_field}"
    return anniversary


def extract_years_at_start(df, date_field):
    integer_years_at_start = pd.Series(
        df[date_field].apply(
            lambda d: relativedelta(period_start, d).years),
        name=f"YearsSince{date_field}",
        index=df.index
    )
    return integer_years_at_start

# for date_field in date_fields:


date_field = "Born"


def split_by_date(df, date_field, period_start):

    pre_df = df.copy()
    post_df = df.copy()
    # experience_start_field = self.experience_start_field
    # experience_end_field = self.experience_end_field
    ann = extract_anniversary(df, date_field, period_start)
    yea = extract_years_at_start(df, date_field)
    pre_df[experience_end_field] = pd.Series(
        ann,
        index=df.index,
        name=experience_end_field
    )
    pre_df[f"YearsSince{date_field}"] = pd.Series(
        yea,
        index=df.index,
    )
    post_df[experience_start_field] = pd.Series(
        ann,
        index=df.index,
        name=experience_start_field
    )
    post_df[f"YearsSince{date_field}"] = pd.Series(
        yea + 1,
        index=df.index,
    )
    output = [pre_df, post_df]
    return output


splits = split_by_date(in_force_df, date_field, period_start)
pd.concat(splits)

# apply split_by_date() recursively for every date_field to get 2^n dataframes
# ... and stitch back together


# %%
