# %%
from llm_details_puller import llm_details_puller
import pandas as pd


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
# get a chunked list of character names
char_list = chunk_my_list(
    corrie_master_df["Character"].to_list(),
    chunk_size=50
)

# check for duplicates
all_corrie_chars = [x for l in char_list for x in l]
assert len(set(all_corrie_chars)) == len(all_corrie_chars)


# %%

# ---------------- run through using gemini ---------------
system_prompt = """
You are a helpful assistant skilled at extracting and formatting character metadata.

You will be given a list of character names from the UK TV show *Coronation Street*. For each character, return a JSON object that best guesses their "gender" which can only be "male" or "female". 
"""

google_llm_class = llm_details_puller(
    model="gemini-2.0-flash",
    model_provider="google_genai",
    system_prompt=system_prompt
)
char_gender_gemini = google_llm_class.collect_chunked_list(
    char_list,
    temperature=0
)

# %%
corrie_gender_fp = "./Data/character_data_gender.csv"
char_gender_gemini.to_csv(corrie_gender_fp)
