# %%
import langchain_core.prompts
import re
import pandas as pd
from datetime import datetime
import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core import prompts
from typing_extensions import List, Optional, Literal
from pydantic import BaseModel, Field

# %%
# defining the structure of the outputs...


class Item(BaseModel):
    input: str = Field(..., description="The original input")
    gender: Literal["male", "female", "unknown"] = Field(
        ..., description="The gender")


class ListItems(BaseModel):
    items: List[Item]

# note instructions on setting up langchain can be found here...
# https://python.langchain.com/docs/tutorials/llm_chain/


class llm_details_puller:
    """A class for posting to and pulling from LLM"""

    def __init__(self,
                 output_class,
                 system_prompt,
                 model="gemini-2.0-flash",
                 model_provider="google_genai"):

        self.model = model
        self.model_provider = model_provider
        self.system_prompt = system_prompt
        self.output_class = output_class

    def change_output_class(self, output_class):
        self.output_class = output_class

    def collect_single_list(self, items_list, temperature=0):
        llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=temperature
        )
        llm_structured = llm.with_structured_output(self.output_class)
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
