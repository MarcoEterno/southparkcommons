
class TaskScheduler:
    def __init__(self, working_memory):
        self.working_memory = working_memory

    def schedule_tasks(self):
        # Sort tasks based on a simple heuristic, e.g., value, then status
        self.working_memory.tasks.sort(key=lambda x: (x.value, x.status))
