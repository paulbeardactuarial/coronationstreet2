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

experience_start_field = "StartDate"
experience_end_field = "ExitDate"
date_fields = ["Born", "FirstAppearance"]
date_field = "Born"

# filter for rows that have experience within period
in_force_df = df[
    (df[experience_start_field] <= period_end) & (
        df[experience_end_field] >= period_start)
].copy()


class ExposureGenerator():
    def __init__(self,
                 df,
                 date_fields,
                 experience_start_field="StartDate",
                 experience_end_field="ExitDate"):
        self.df = df
        self.date_fields = date_fields
        self.experience_start_field = experience_start_field
        self.experience_end_field = experience_end_field

    def set_df(self, df):
        self.df = df

    def set_date_fields(self, date_fields):
        self.date_fields = date_fields

    def extract_anniversary(self, date_field, period_start):
        month = self.df[date_field].dt.month
        day = self.df[date_field].dt.day
        is_feb_29 = (month == 2) & (day == 29)
        anniv_year = pd.Series(period_start.year, index=self.df.index)
        anniversary = pd.to_datetime({
            'year': anniv_year,
            'month': month,
            'day': day
        }, errors='coerce')
        anniversary[anniversary.isna() & is_feb_29] = pd.to_datetime(
            {
                'year': anniv_year[anniversary.isna()],
                'month': 2,
                'day': 28
            }
        )
        anniversary = anniversary.where(
            anniversary >= period_start,
            anniversary + pd.DateOffset(years=1)
        )
        anniversary.name = f"Anniv{date_field}"
        return anniversary

    def extract_years_at_start(self, date_field, period_start):
        integer_years_at_start = pd.Series(
            self.df[date_field].apply(
                lambda d: relativedelta(period_start, d).years),
            name=f"YearsSince{date_field}",
            index=self.df.index
        )
        return integer_years_at_start

    def initialise_period_df(self, period_start):
        df = self.df
        period_end = pd.Timestamp(datetime.date(
            period_start.year+1,
            period_start.month,
            period_start.day)
        )
        in_force_df = df[
            (df[self.experience_start_field] <= period_end) & (
                df[self.experience_end_field] >= period_start)
        ].copy()
        in_force_df["PeriodStartDate"] = period_start
        in_force_df["PeriodEndDate"] = period_end
        return in_force_df

    def carve_by_date(self, period_df, date_field, period_start):
        new_field_name = f"YearsSince{date_field}"
        pre_df = period_df.copy()
        post_df = period_df.copy()
        ann = self.extract_anniversary(date_field, period_start)
        yea = self.extract_years_at_start(date_field, period_start)
        pre_df["PeriodEndDate"] = pd.Series(
            ann,
            name="PeriodEndDate"
        )
        pre_df[new_field_name] = pd.Series(yea)
        post_df["PeriodStartDate"] = pd.Series(
            ann,
            name="PeriodStartDate"
        )
        post_df[new_field_name] = pd.Series(yea + 1)
        output = pd.concat([pre_df, post_df])
        new_index = [*period_df.index.names, new_field_name]
        output = output.reset_index().set_index(new_index)
        return output

    def carve_by_dates(self, period_df, period_start):
        for date_field in self.date_fields:
            period_df = self.carve_by_date(period_df, date_field, period_start)
        return period_df

    def clip_period_dates(self, df):
        for field in ["PeriodStartDate", "PeriodEndDate"]:
            df[field] = df[field].clip(
                lower=df[experience_start_field],
                upper=df[experience_end_field]
            )
        return df

    def assign_exposure(self, df):
        df = df.assign(
            Exposure=((df["PeriodEndDate"]-df["PeriodStartDate"]).days)/365.25
        )
        return df

    def construct_exposure_single_period(self, period_start):
        period_df = self.initialise_period_df(period_start)
        period_df = self.carve_by_dates(period_df, period_start)
        period_df = self.clip_period_dates(period_df)
        # period_df = self.assign_exposure(period_df)
        return period_df


# for date_field in date_fields:
eg = ExposureGenerator(in_force_df, ["Born", "FirstAppearance"])


df = eg.construct_exposure_single_period(period_start)
# (df["PeriodEndDate"]-df["PeriodStartDate"]).dt.days/365.25


# %%
date_field = "Born"


splits = carve_by_date(in_force_df, date_field, period_start)
exp_table = pd.concat(splits)
exp_table

# clip the start and end dates based on...
# 1. period start/end
# 2. row start/end of initial data

# then join it to exp_table and clip that!


# apply carve_by_date() recursively for every date_field to get 2^n dataframes
# ... and stitch back together
for field in [experience_start_field, experience_end_field]:
    exp_table[field] = exp_table[field].clip(
        lower=period_start,
        upper=period_end
    )

# %%
