# %%
from pydantic import BaseModel, Field
from typing_extensions import List, Optional, Literal
from llm_details_puller import llm_details_puller
import pandas as pd

# The purpose of this script is to read in `character_data.csv`...
# ...and enrich it using interactions with LLMs

# The script has two LLM sections that do the following:
# 1. Extract "gender" based on "Character"
# 2. Extract "sic" and "sec" based on "Occupation"

# I have opted to use Gemini API for the LLM because it is free :)

# %%
corrie_data_fp = "./Data/character_data.csv"
corrie_master_df = pd.read_csv(corrie_data_fp)
corrie_master_df = corrie_master_df.pivot(
    columns="Field", values="Value", index="Character").reset_index()


# %%
def chunk_my_list(list_to_be_chunked, chunk_size):
    no_items = len(list_to_be_chunked)
    no_chunks = no_items//chunk_size + 1
    master_list = []
    for i in range(no_chunks):
        mini_list = list_to_be_chunked[(
            chunk_size * i):(chunk_size * i + chunk_size)]
        master_list.append(mini_list)
    return (master_list)


# %%

# ================================================
# ====== get gender of each character name =======
# ================================================

# get a chunked list of character names
char_list = chunk_my_list(
    corrie_master_df["Character"].to_list(),
    chunk_size=50
)

# check for duplicates
all_corrie_chars = [x for l in char_list for x in l]
assert len(set(all_corrie_chars)) == len(all_corrie_chars)

# %%


class Item(BaseModel):
    input: str = Field(..., description="The original input")
    gender: Literal["male", "female", "unknown"] = Field(
        ..., description="The gender")


class ListItems(BaseModel):
    items: List[Item]


system_prompt = """
You are a helpful assistant skilled at extracting and formatting character metadata.

You will be given a list of character names from the UK TV show *Coronation Street*. For each character estimate their gender.
"""

google_llm_class = llm_details_puller(
    output_class=ListItems,
    system_prompt=system_prompt,
    model="gemini-2.0-flash",
    model_provider="google_genai"
)
char_gender_gemini = google_llm_class.collect_chunked_list(
    char_list,
    temperature=0
)

# %%
corrie_gender_fp = "./Data/character_data_gender.csv"
char_gender_gemini.to_csv(corrie_gender_fp)


# %%

# ================================================
# ========== get SEC of each profession ==========
# ================================================

class ItemOccupation(BaseModel):
    input: str = Field(..., description="The original input")
    sec: Literal[
        "High",
        "Medium",
        "Low"
    ] = Field(
        ..., description="The socioeconomic classification")
    sic: Literal[
        "Agriculture, forestry & fishing",
        "Mining, energy and water supply",
        "Manufacturing",
        "Construction",
        "Wholesale, retail & repair of motor vehicles",
        "Transport & storage",
        "Accommodation & food services",
        "Information & communication",
        "Financial & insurance activities",
        "Real estate activities",
        "Professional, scientific & technical activities",
        "Administrative & support services",
        "Public admin & defence; social security",
        "Education",
        "Human health & social work activities",
        "Other services"] = Field(
        ..., description="The standard industrial classification")


class ListItemsOccupation(BaseModel):
    items: List[ItemOccupation]


# get a chunked list
all_corrie_occupation = corrie_master_df["Occupation"].dropna(
).drop_duplicates().to_list()

occupation_list = chunk_my_list(
    all_corrie_occupation,
    chunk_size=50
)

# check for duplicates
assert len(set(all_corrie_occupation)) == len(all_corrie_occupation)

# %%

SystemPromptIndustrial = """
You are a helpful assistant skilled at extracting and formatting British occupation metadata.

You will be given a list of occupations of people living in England. For each one:
- guess the socioeconomic class of the person with that occupation in one of 3 categories, as either "high", "mid" or "low".
- guess the "standard industrial classification" of the occupation
"""

google_llm_class = llm_details_puller(
    model="gemini-2.0-flash",
    model_provider="google_genai",
    system_prompt=SystemPromptIndustrial,
    output_class=ListItemsOccupation
)
occupation_info_gemini = google_llm_class.collect_chunked_list(
    occupation_list,
    temperature=0
)

# %%
corrie_occ_fp = "./Data/occupation_data_sec_sic.csv"
occupation_info_gemini.to_csv(corrie_occ_fp)

# %%

corrie_master_df = corrie_master_df.merge(
    occupation_info_gemini,
    how="left",
    left_on="Occupation",
    right_index=True
)

corrie_master_df = corrie_master_df.merge(
    char_gender_gemini,
    how="left",
    left_on="Character",
    right_index=True
)

corrie_master_df["gender"] = corrie_master_df["gender"].str.capitalize()

# %%
corrie_master_df = corrie_master_df.rename(str.capitalize, axis=1)
corrie_master_df = corrie_master_df.melt(
    id_vars="Character",
    var_name="Field",
    value_name="Value").sort_values(by="Character").dropna(how="any").set_index("Character")

corrie_data_fp = "./Data/character_data_enriched.csv"
corrie_master_df.to_csv(corrie_data_fp)
