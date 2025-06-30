# %%
import pandas as pd
import numpy as np
import datetime

# %%
corrie_segments = pd.read_csv("./Data/character_data_segmented.csv")
corrie_segments = corrie_segments[~corrie_segments["Born"].isna(
)].reset_index().drop("index", axis=1)

# convert to datetime format
date_fields = ["Born", "Died", "First appearance",
               "Last appearance", "Exit date", "Start date"]
corrie_segments.loc[:, date_fields] = corrie_segments.loc[:,
                                                          date_fields].apply(pd.to_datetime)


# %%

# drop those without "Born" day

# corrie_segments["exit_date"].apply(lambda x: datetime.datetime.strptime(x, format = 'd-m-Y'))

# applying datetime.datetime.strptime()
# pd.dt.to_timestamp()
# pd.dt.to_period()


# .dt.to_timestamp()
# corrie_segments["Born"].year
