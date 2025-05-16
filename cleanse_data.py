# %%
import pandas as pd
import numpy as np
import datetime
import re

# %%
corrie_data_fp = "./Data/character_data.csv"
corrie_data = pd.read_csv(corrie_data_fp)


# %%
c_life_death = corrie_data[corrie_data["Field"].isin(
    ["Born", "Died", "First appearance", "Last appearance", "Duration"])]
c_life_death = c_life_death.pivot(
    columns="Field", values="Value", index="Character").reset_index()
c_life_death

# %%
date_string_series = c_life_death["Born"]

full_date_pattern = r'^\s*\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$'

date_strings = date_string_series.loc[
    date_string_series.str.match(full_date_pattern).fillna(False)
]

# end_year_regex = r'(\d{4})\s*$'
# mid_month_regex = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
# start_day_regex = r'^\s*(\d{1,2})'

# year = date_strings.str.extract(end_year_regex)[0]
# month = date_strings.str.extract(mid_month_regex)[0]
# day = date_strings.str.extract(start_day_regex)[0]

date_strings = date_strings.str.replace("st|nd|rd|th", "", n=1, regex=True)
date_strings = date_strings.apply(
    lambda x: datetime.datetime.strptime(x, "%d %B %Y")
)
date_strings
# %%
