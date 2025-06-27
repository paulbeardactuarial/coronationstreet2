# %%
import json
from langchain_core import prompts
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
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
you are a helpful chatbot that is great at finding contact details. 
You will be given a list of character names from the UK TV show Coronation Street. Return a JSON object only. 
For each character you will predict the "gender" and "age" and add them as properties of that character. This will be returned in JSON format. 
The gender must be either "male" or "female". No other gender is allowed. Return the JSON only. The JSON object MUST have the same number of items as the number of characters you were given as input.
"""

system_prompt_detailed = """
You are a helpful assistant skilled at extracting and formatting character metadata.

You will be given a list of character names from the UK TV show *Coronation Street*. For each character, return a JSON object that includes their "gender" 
"""

google_llm_class = llm_details_puller(
    model="gemini-2.0-flash",
    model_provider="google_genai",
    system_prompt=system_prompt_detailed
)
char_gender_gemini = google_llm_class.collect_chunked_list(
    char_list[:3],
    temperature=0
)

o = google_llm_class.collect_single_list(
    char_list[2],
    temperature=0
)


# %%
google_llm_class.collect_single_list(char_list[0])


# %%

model = init_chat_model(
    model="gemini-2.0-flash",
    model_provider="google_genai"
)
prompt_template = prompts.ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("user", "{list_of_names}")
    ]
)
prompt = prompt_template.invoke({"list_of_names": char_list[:1]})
response = model.invoke(prompt)

# %%


def llm_response_to_json(content):
    output = content
    output = output.removeprefix("```json")
    output = output.removesuffix("```")
    return output


json_response = llm_response_to_json(response.content)
list_response = json.loads(json_response)
df = pd.DataFrame(list_response, index=char_list[:1]).reset_index(
    names="item_name")

# %%
