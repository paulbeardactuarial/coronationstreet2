# %%
import pandas as pd
from datetime import datetime

# %%
corrie_master_df_clean = (
    pd.read_csv("./Data/character_data_segmented.csv")
    # .assign(YearsOnSt = lambda x: x["Start date"] - x["Exit date"])
    .query("Segment == `Max segment`")
    .set_index("Character")
    .assign(Returning=lambda x: x.Segment > 1)
)

# %%
pd.read_csv("./Data/character_data_segmented.csv")


date_cols = ['Born', 'Died', 'Exit date', 'Start date',
             'First appearance', 'Last appearance']


df = pd.read_csv("./Data/character_data_segmented.csv")

df.loc[:, date_cols] = df.apply(
    {col: lambda x: pd.to_datetime(x) for col in date_cols}
)


def years_diff(start_series, end_series):
    time_delta = end_series - start_series
    years_diff_series = time_delta.apply(lambda x: round(x.days/365.25, 2))
    return years_diff_series


df["YearsOnStreet"] = years_diff(df["Start date"], df["Exit date"])
df["AgeEnterStreet"] = years_diff(df["Born"], df["First appearance"])
df["AgeLeftStreet"] = years_diff(df["Born"], df["Last appearance"])


# %%
sum_y_o_s = df.groupby("Character")["YearsOnStreet"].transform(sum)
(df
 .query("Segment == `Max segment`")
 .assign(Returner=lambda x: x['Max segment'] > 1)
 .drop(columns=[
     "YearsOnStreet",
     "Segment",
     "Max segment"]
 )
 .merge(sum_y_o_s, how="inner", left_index=True, right_index=True)
 )


# %%
