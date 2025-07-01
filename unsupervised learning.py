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

# df["YearsOnStreet"] =
pd.to_numeric(df["Start date"] - df["Exit date"])/365.25


# %%
