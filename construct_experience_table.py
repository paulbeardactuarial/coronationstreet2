# %%
import pandas as pd
from ExposureGenerator import ExposureGenerator

input_fp = "./Data/character_data_segmented.parquet"
df = pd.read_parquet(input_fp)

# drop those without "Born" day
df = df[~df["Born"].isna()]

# for date_field in date_fields:
eg = ExposureGenerator(
    df,
    ["StartDate", "FirstAppearance", "Born"]
)

output = eg.construct_exposure_full()
output = output[output["Exposure"] > 0]
output_fp = "./Data/exposure_table.parquet"
output.to_parquet(output_fp, engine="pyarrow")

# %%
