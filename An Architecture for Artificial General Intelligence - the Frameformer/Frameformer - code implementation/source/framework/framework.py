from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os
from langchain.llms import OpenAI
import openai

from ..utils.function_utils import return_number_of_calls
from ..llm.llm_functions import OpenAI_count_calls
from ..root.apikey import API_KEY as apikey
from ..sandbox.sandbox import execute_program_routine_python

# initializing the language model parameters
os.environ["OPENAI_API_KEY"] = apikey
openai.api_key = apikey  # line to maintain only if the llm is called by openai function without langchain

# default_llm = OpenAI_count_calls(temperature=0.9, model="gpt-4-1106-preview")

return_number_of_calls.calls = 0

# TODO: refactor this class separating the creation of new frameworks from the execution of the framework


@dataclass
class Framework:
    id: int = field(default=return_number_of_calls())  # generate unique id for each framework.
    name: str = ""  # generate unique name for each framework.
    objective: str = ""
    text_routine: str = ""
    program_routine: str = ""
    program_instructions: str = ""  # instructions to run the program routine: how to set parameters,
    # which debug process to use, etc
    knowledge_of_task: List[str] = field(default_factory=list)  # contains the info gathered by running the
    # framework multiple times (code errors, task solved in how many attempts, etc)
    parent_frameworks: List[str] = field(default_factory=list)
    child_frameworks: List[str] = field(default_factory=list)

    @staticmethod
    def determine_optimal_framework_type(objective: str) -> str:
        """
        Determines the optimal framework type to use for the task.

        Args:
            objective: Objective to determine the optimal framework type for.

        Returns:
            str: Optimal framework type to use for the task. (text or program)
        """
        # TODO: substitute with llm call
        framework_type = "text" #to substitute with the llm call below

        """default_llm(
            prompt=f"You are a discriminator. You need to assess whether a given task "
                   f"will be best accomplished by writing the instructions of that task "
                   f"and passing them to a language model, or instead writing a computer"
                   f"program and execute it. You must reply with only the word text if "
                   f"the task requires text instructions, and only the word program if "
                   f"it is appropriate to write a computer program. For example:"
                   f"\nObjective: predicting the next chess move given a chess board. \n"
                   f"Output: program"
                   f"\nObjective: summarize a given sentence. \n"
                   f"Output: text"
                   f"\nObjective: {objective}\n"
                   f"Output: ")"""

        return framework_type

    def build_text_routine(self, objective, resource_budget_in_euros = 0.1) -> None:
        """
        Builds the text routine of the framework following an iterative process.

        Returns:
            str: Text routine of the framework.
        """
        framework_is_not_complete = True
        resources_are_not_exhausted = True
        while framework_is_not_complete and resources_are_not_exhausted:
            #Build a better text routine

            #Check if the framework is complete
            framework_is_not_complete = False#TODO: substitute with llm call

            #Check if the resources are exhausted
               #calculate resources
            resources =  0 # TODO calculation of resources (use config/api_pricing.py)
            resources_are_not_exhausted = resources > resource_budget_in_euros



    def build_program_routine(self, objective, resource_budget_in_euros = 0.1) -> None:
        """
        Builds the program routine of the framework.

        Returns:
            str: Program routine of the framework.
        """

        framework_is_not_complete = True
        resources_are_not_exhausted = True
        while framework_is_not_complete and resources_are_not_exhausted:
            #Build a better program routine

            #Check if the framework is complete
            framework_is_not_complete = False

            #Check if the resources are exhausted
                #calculate resources
            resources =  # TODO calculation of resources (use config/api_pricing.py)
            resources_are_not_exhausted = resources > resource_budget_in_euros

    def execute_text_routine(self, objective: str, resource_budget_in_euros: float = 0.1) -> Tuple[str, str]:
        """
        Runs the text routine of the framework.

        Args:
            objective: Objective to run the text routine for.
            resource_budget_in_euros: Resource budget to use for the text routine.

        Returns:
            Tuple[str, str]: Output and error of the text routine.
        """
        # decide which llm to use based on the budget, than call it an return the answer.
        # TODO: write the  llm call

    def execute_program_routine(self, objective: str, resource_budget_in_euros: float = 0.1) -> Tuple[str, str]:
        """
        Runs the program routine of the framework.

        Args:
            objective: Objective to run the program routine for.
            resource_budget_in_euros: Resource budget to use for the program routine.

        Returns:
            Tuple[str, str]: Output and error of the program routine.
        """
        # Execute the program routine
        try:
            output, error = execute_program_routine_python(script_name=self.program_routine)
            #TODO: implement real security features in sandbox
            return output, error

        except Exception as e:
            print(f"An error occurred while executing the program routine of framework: {e}")
            return "",str(e)

    def debug_program_routine(self, objective: str) -> None:
        """
        Debugs the program routine of the framework.

        Args:
            objective: Objective to debug the program routine for.
        """
        pass


    @staticmethod
    def create_new_framework(objective: str, name : str="", text_routine:str = "", program_routine: str = "",
                             program_instructions: str ="") -> object:
        """
        Creates a new framework that can be used to solve a task.
        :param name: Name of the framework.
        :param objective: Description of the task the framework should solve.
        :param text_routine: Text routine of the framework.
        :param program_routine: Program routine of the framework.
        :param program_instructions: Instructions to run the program routine: how to set parameters,

        """
        framework = Framework(name=name, objective=objective, text_routine=text_routine,
                              program_routine=program_routine, program_instructions=program_instructions)

        # Determine whether the framework is best built using text instructions or a program routine
        framework_type = framework.determine_optimal_framework_type(objective=objective)
        # If text instructions, then the framework is built using the text routine
        if framework_type == "text":
            framework.build_text_routine(objective=objective)
            return framework
        # If program routine, then the framework is built using the program routine
        elif framework_type == "program":
            framework.build_program_routine(objective=objective)
            return  framework
        # If neither, then the framework is not built
        else:
            raise ValueError(f"Framework type not recognized: {framework_type}")
    def update_information(self, **kwargs) -> None:
        """
        Updates the information of the desired fields of a framework.

        Args:
            **kwargs: Information to update.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_information(self, **kwargs) -> None:
        """
        Adds information to the desired fields of a framework.
        Args:
            **kwargs: Information to add.
        """
        for key, value in kwargs.items():
            getattr(self, key).append(value)

    def remove_information(self, **kwargs) -> None:
        """
        Removes information from the desired fields of a framework.

        Args:
            **kwargs: Information to remove.
        """
        for key, value in kwargs.items():
            try:
                getattr(self, key).remove(value)
            except ValueError:
                raise ValueError(f"Value '{value}' not found in '{key}'.")

    def get_information(self, *args) -> Dict[str, List[str]]:
        """
        Gets the information of the desired fields of a framework.

        Args:
            *args: Fields to get information from.

        Returns:
            Dict[str, List[str]]: Information of the desired fields of a framework.
        """
        information = {}
        for arg in args:
            information[arg] = getattr(self, arg)
        return information

    def to_json(self) -> Dict[str, List[str]]:
        """
        Converts the framework to a JSON object.

        Returns:
            Dict[str, List[str]]: JSON object representing the framework.
        """
        jsoneable_framework = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                jsoneable_framework[key] = value
        return jsoneable_framework

    def __str__(self) -> str:
        return f"Framework: {self.name}, Description: {self.objective}"

    def __repr__(self) -> str:
        string=""
        for key, value in self.__dict__.items():
            string+=f"{key}: {value}\n"
        return string