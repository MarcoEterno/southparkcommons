from apikey import API_KEY as apikey
from prompts import QA_format
from utils import count_calls
from AutoGPT.source.memory.working_memory import WorkingMemory

import os
import streamlit as st
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = apikey


@count_calls
def OpenAI_count_calls(*args, **kwargs):
    return OpenAI(*args, **kwargs)


default_llm = OpenAI_count_calls(temperature=0.9, model="gpt-3.5-turbo-instruct")

# App framework
st.title("AutoGPT")

# initialize the components
working_memory = WorkingMemory()

# generate first task list

# generate first thought

# organize


process_is_running = True
while process_is_running:
    # 1) fetch user input
    model_question = "abc"
    user_input = st.text_input("Answer", "You should do that")
    # x = default_llm(prompt=user_input)
    if user_input != "":
        # update model state by putting user_input in the context window
        new_info = QA_format(model_question, user_input)
    # 2) update the state of the system based on the user input, and execute next task

    # 3) return output to screen if user input is required
    model_output = ""
    st.write(model_output)

    # 4) stop the execution when there are no tasks left
    if working_memory.tasks == []:
        process_is_running = False

# 5) return the final output to the screen
st.write("Done. here is the final output: \n")