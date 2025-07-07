# %%
import pyarrow
import pandas as pd
from datetime import datetime
from cleaner_functions import clean_column_names, years_diff

input_fp = "./Data/character_data_segmented.parquet"
output_fp = "./Data/character_data_clean.parquet"


df = pd.read_parquet(input_fp)

# calculate a series for the years difference between two date series
df["AgeEnterStreet"] = years_diff(df["Born"], df["FirstAppearance"])
df["AgeLastOnStreet"] = years_diff(df["Born"], df["ExitDate"])
df["YearsOnStreet"] = years_diff(df["StartDate"], df["ExitDate"])

sum_y_o_s = df.groupby("Character")["YearsOnStreet"].transform(sum)
df = (df
      .query("Segment == MaxSegment")
      .assign(Returner=lambda x: x['MaxSegment'] > 1)
      .drop(columns=[
          "YearsOnStreet",
          "Segment",
          "MaxSegment"]
      )
      .merge(sum_y_o_s, how="inner", left_index=True, right_index=True)
      .assign(AppearPerYear=lambda x: x['NumberOfAppearances']/x['YearsOnStreet'])
      )

df = df.convert_dtypes()
df = df.set_index("Character")

# %%
# Save DataFrame as Parquet with metadata (data date: 30/06/25)

# Define column metadata (update descriptions as needed)
column_metadata = {
    "Gender": "Character gender (M/F)",
    "NumberOfAppearances": "Total number of appearances",
    "FirstAppearance": "Datetime of first appearance",
    "LastAppearance": "Datetime of last appearance",
    "ExitStatus": "Character's exit status (Alive/Death/Exit)",
    "Born": "Datetime of character's birth (if known)",
    "Died": "Datetime of character's death (if occcurred)",
    "AgeEnterStreet": "Age at first appearance (years)",
    "AgeLastOnStreet": "Age at exit (years)",
    "YearsOnStreet": "Total years on the show, accounting for gaps",
    "Returner": "Boolean, True if character returned after exit",
    "NumberTimesMarried": "Number of times character married",
    "NumberChildren": "Number of children character has",
    "Occupation": "Character's occupation",
    "Industry": "Industry sector of occupation",
    "BigamyCommitted": "Whether character was involved in bigamous marriage (boolean)"
}

df = df[list(column_metadata.keys())]

# Add metadata to DataFrame
df.attrs['column_metadata'] = column_metadata
df.attrs['data_date'] = "2025-06-30"

# Save to Parquet
df.to_parquet(output_fp, engine="pyarrow")

# %%
