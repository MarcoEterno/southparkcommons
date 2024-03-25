from apikey import API_KEY as apikey
from  AutoGPT.source.llm.llm_functions import OpenAI_count_calls


from collections import deque
import os
from langchain.llms import OpenAI

default_llm = OpenAI_count_calls(temperature=0.9, model="gpt-3.5-turbo-instruct")


class WorkingMemory:

    def __init__(self):
        self.tasks = deque()
        self.thoughts = deque()

    def add_thought(self, thought):
        self.thoughts.append(thought)

    def extract_relevant_thoughts(self):
        # call the llm to extract the relevant thoughts
        default_llm(prompt="abc")


    def add_task(self, task):
        self.tasks.append(task)

    def reorder_tasks_based_on_importance(self):
        # call the llm to get the importance of each task, then reorder the tasks based on it and on the thoughts.
        pass

    def get_next_task(self):
        return self.tasks.popleft()

    def generate_thought_based_on_tasks(self):
        # call the llm to generate a thought based on the tasks
        pass