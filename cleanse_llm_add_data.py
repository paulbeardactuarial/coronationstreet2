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
            chunk_size * i):(chunk_size * i + chunk_size-1)]
        master_list.append(mini_list)
    return (master_list)


# %%
# get a chunked list of character names
char_list = chunk_my_list(
    corrie_master_df["Character"].to_list(),
    50
)


# %%

# ---------------- run through using gemini ---------------
system_prompt = """
you are a helpful chatbot that is great at finding contact details. You will be given a list of character names from the UK TV show Coronation Street. Predict the gender of each character and return in JSON format. Return only for each character a gender of either "male" or "female". No other gender is allowed. Return the JSON only. The JSON object MUST have the same number of items as the number of characters you were given as input.
"""

google_llm_class = llm_details_puller(
    model="gemini-2.0-flash",
    model_provider="google_genai",
    system_prompt=system_prompt
)
char_gender_gemini = google_llm_class.collect_chunked_list(
    char_list)

# %%
