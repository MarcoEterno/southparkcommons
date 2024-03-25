from langchain.schema import AIMessage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import nltk
import os

# Download the punkt tokenizer (if not already downloaded)
nltk.download('punkt')

def open_file_get_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            file.close()
            return text
    except:
        raise Exception("Error: file not found")

def count_tokens(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokens = nltk.word_tokenize(text)
            file.close()
            return len(tokens)
    except:
        raise Exception("Error: file not found")

def count_tokens_of_string(text):
    tokens = nltk.word_tokenize(str(text))
    return len(tokens)


"""
in context memory management that uses RAG from the worldview + summarization to fit all useful info in the context length
"""



def summarize_infos_relevant_to_task(task, text_to_summarize, default_llm, context_length=8000):
    """Summarizes useful info"""

    """First step: counting the context length"""
    num_tokens = count_tokens_of_string(text_to_summarize)

    """Second step: reads all the world_model and summarizes the relevant info"""
    iterations = num_tokens // context_length + 1
    summarization = []

    for i in range(iterations):
        prompt = "YOUR TASK IS: " + str(task) + " READ THIS WORLD MODEL WRITTEN BY YOU AND OUTPUT" + \
                 "ALL THE RELEVANT INFORMATION TO SOLVE THE TASK. YOU ARE NOW SUMMARIZING THE " + str(i) + " / " + \
                 str(iterations) + " PART OF THE WORLD MODEL, WHICH FOLLOWS: " + str(text_to_summarize)
        summarization.append(default_llm(prompt) + '\n')

    """Third step"""
    num_summarizations_done = 1
    while count_tokens_of_string(summarization) > context_length and num_summarizations_done < 3:
        # first argue that the summarization is too long, we define a string, and pass it to the function.
        summarize_infos_relevant_to_task(str(task) + "Also: YOU HAVE ALREADY SUMMARIZED THE INFO: BUT IT WAS TOO LONG: " + \
                                                str(count_tokens_of_string(str(summarization))) + " INSTEAD OF THE CONTEXT LENGTH " + str(context_length) + \
                                                " NOW TRY TO SUMMARIZE IT IN LESS TOKENS!", text_to_summarize, default_llm, context_length)
        num_summarizations_done += 1

    return summarization

def give_answer(task, relevant_text, default_llm, context_length = 8000):
    """Uses the relevant info to give an answer to the task """

    """First step: counting the context length"""
    num_tokens = count_tokens_of_string(relevant_text)
    
    if num_tokens > context_length:
        relevant_text = summarize_infos_relevant_to_task(task, relevant_text, default_llm, context_length)

    """Second step: gives the answer required by the task after having read the task and the relevant text"""
    prompt = "YOUR TASK IS: " + str(task) + " READ AND USE THE FOLLOWING RELEVANT INFORMATION TO SOLVE THE TASK. " + \
             "BE BRIEF AND CONCISE: " + str(relevant_text)
    return default_llm(prompt)


def determine_possible_question_to_user():
    """
    determine whether user input is required and returns the question.
    """
    pass
