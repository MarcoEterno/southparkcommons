
class Agent:
    def __init__(self, working_memory):
        self.working_memory = working_memory

    def execute_task(self, task):
        # Example logic for task execution (can be expanded)
        print(f"Executing task {task.id}: {task.name}")
        task.update_status("in_progress")

        # Task execution logic here...

        task.update_status("closed")
        print(f"Task {task.id} completed.")
