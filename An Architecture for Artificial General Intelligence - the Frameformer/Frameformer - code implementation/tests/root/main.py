
from AutoGPT.source.agent.agent import Agent
from AutoGPT.source.task_management.working_memory import WorkingMemory
from AutoGPT.source.task_management.task import Task

def main():
    working_memory = WorkingMemory()
    agent = Agent(working_memory)

    # Example task
    task = Task(1, "Sample Task", "Description of the task", "open", "high", [])
    working_memory.add_task(task)
    agent.execute_task(task)

if __name__ == "__main__":
    main()
