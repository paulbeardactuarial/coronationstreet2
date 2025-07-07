# %%
import pyarrow
import pandas as pd
from datetime import datetime

input_fp = "./Data/character_data_segmented.csv"
output_fp = "./Data/character_data_clean.csv"

df = pd.read_csv(input_fp)

# calculate a series for the years difference between two date series


def years_diff(start_series, end_series):
    time_delta = end_series - start_series
    years_diff_series = time_delta.apply(lambda x: round(x.days/365.25, 2))
    return years_diff_series


date_cols = ['Born', 'Died', 'Exit date', 'Start date',
             'First appearance', 'Last appearance']

df.loc[:, date_cols] = df.apply(
    {col: lambda x: pd.to_datetime(x) for col in date_cols}
)

df["YearsOnStreet"] = years_diff(df["Start date"], df["Exit date"])
df["AgeEnterStreet"] = years_diff(df["Born"], df["First appearance"])
df["AgeLastOnStreet"] = years_diff(df["Born"], df["Exit date"])

sum_y_o_s = df.groupby("Character")["YearsOnStreet"].transform(sum)
df = (df
      .query("Segment == `Max segment`")
      .assign(Returner=lambda x: x['Max segment'] > 1)
      .drop(columns=[
          "YearsOnStreet",
          "Segment",
          "Max segment"]
      )
      .merge(sum_y_o_s, how="inner", left_index=True, right_index=True)
      .assign(AppearPerYear=lambda x: x['Number of appearances']/x['YearsOnStreet'])
      )

df = df.convert_dtypes()
df = df.set_index("Character")

# clean up final columns...
df = df.drop(
    ['Birthplace',
     'Duration',
     'Children',
     'Father',
     'Mother',
     'Sibling(s)',
     'Spouse(s)',
     'Played by',
     'Residence',
     'Start date',
     'Exit date',
     'AppearPerYear',
     'Sec'],
    axis=1
)

# %%
df = df.rename(
    {"Sic": "Industry"},
    axis=1
)


# %%
def clean_column_names(columns):
    """
    Clean column names with consistent capitalization, spacing, and terminology.

    Rules applied:
    1. PascalCase (each word starts with capital letter)
    2. Remove spaces (words separated by capitals)
    3. Standardize "No" to "Number"

    Args:
        columns: Index or list of column names to clean

    Returns:
        List of cleaned column names
    """
    cleaned_names = []

    for col in columns:
        col_str = str(col)

        # If column has spaces, process it
        if ' ' in col_str:
            words = col_str.split()

            # Process each word
            cleaned_words = []
            for word in words:
                # Capitalize first letter, lowercase the rest
                cleaned_word = word.capitalize()

                # Replace "No" with "Number"
                if cleaned_word == "No":
                    cleaned_word = "Number"

                cleaned_words.append(cleaned_word)

            # Join without spaces (PascalCase)
            cleaned_name = "".join(cleaned_words)
        else:
            # Already in PascalCase format, just handle "No" replacement
            cleaned_name = col_str
            # Replace standalone "No" at word boundaries with "Number"
            import re
            cleaned_name = re.sub(r'\bNo\b', 'Number', cleaned_name)

        cleaned_names.append(cleaned_name)

    return cleaned_names

# %%
# final column order


column_order = [
    # Basic demographics - fundamental character info
    'Gender',
    'NumberOfAppearances',

    # Show timeline - when they appeared
    'FirstAppearance',
    'LastAppearance',
    'ExitStatus',
    'Born',
    'Died',

    # Age-related metrics - derived from timeline
    'AgeEnterStreet',
    'AgeLastOnStreet',
    'YearsOnStreet',

    # Show participation metrics
    'Returner',

    # Personal life characteristics
    'NumberTimesMarried',
    'NumberChildren',

    # Professional life
    'Occupation',
    'Industry',

    # Special/unusual characteristics
    'BigamyCommitted'
]

df.columns = clean_column_names(df.columns)
df = df[column_order]
df.to_csv(output_fp)


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

# Add metadata to DataFrame
df.attrs['column_metadata'] = column_metadata
df.attrs['data_date'] = "2025-06-30"

# Save to Parquet
df.to_parquet("./Data/character_data_clean.parquet", engine="pyarrow")
