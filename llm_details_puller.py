# %%
from openai import OpenAI
import langchain_core.prompts
import re
import pandas as pd
from datetime import datetime
import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core import prompts
import json
from typing_extensions import List, Optional, Literal
from pydantic import BaseModel, Field

# %%
# defining the structure of the outputs...


class Item(BaseModel):
    input: str = Field(..., description="The original input")
    gender: Optional[Literal["male", "female"]] = Field(
        None, description="The gender")


class ListItems(BaseModel):
    items: List[Item]

# note instructions on setting up langchain can be found here...
# https://python.langchain.com/docs/tutorials/llm_chain/


class llm_details_puller:
    """A class for posting to and pulling from LLM"""

    def __init__(self,
                 model="gemini-2.0-flash",
                 model_provider="google_genai",
                 system_prompt="""
you are a helpful chatbot that is great at finding contact details. You will be given a list of UK-based IELTS schools. Find the contact details of each and return in JSON format. Return only for each item the "phone", "email" and "address" fields populated for that school. Return the JSON only. The JSON object MUST have the same number of items as the number of schools you were given as input.
"""):

        self.model = model
        self.model_provider = model_provider
        self.system_prompt = system_prompt

    def collect_single_list(self, items_list, temperature=0):
        llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=temperature
        )
        llm_structured = llm.with_structured_output(ListItems)
        prompt_template = prompts.ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", "{list_of_items}")
            ]
        )
        prompt = prompt_template.invoke(
            {"list_of_items": items_list}
        )
        output = llm_structured.invoke(prompt)
        df = pd.DataFrame(output.dict()["items"]).set_index("input")
        return df

    def collect_chunked_list(self, items_list, temperature=0):
        output_dfs = []
        i = 0
        for chunk in items_list:
            i = i + 1
            print(f"collecting for chunk {i} of {len(items_list)}")
            df = self.collect_single_list(chunk, temperature=temperature)
            output_dfs.append(df)

        details_complete = pd.concat(output_dfs)

        return details_complete
# %%
