from ..root.apikey import API_KEY as apikey
import os
from langchain.llms import OpenAI
import openai

# initializing the language model parameters
os.environ["OPENAI_API_KEY"] = apikey
openai.api_key = apikey

from ..utils.function_utils import count_calls, return_number_of_calls


@count_calls
def OpenAI_count_calls(*args, **kwargs):
    return OpenAI(*args, **kwargs)

