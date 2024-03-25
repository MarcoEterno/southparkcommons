
class WorkingMemory:
    def __init__(self):
        self.tasks = []
        self.thoughts = []

    def add_task(self, task):
        self.tasks.append(task)

    def add_thought(self, thought):
        self.thoughts.append(thought)

    def find_task_by_id(self, task_id):
        return next((task for task in self.tasks if task.id == task_id), None)

    def remove_task(self, task_id):
        task_to_remove = self.find_task_by_id(task_id)
        if task_to_remove:
            self.tasks.remove(task_to_remove)
