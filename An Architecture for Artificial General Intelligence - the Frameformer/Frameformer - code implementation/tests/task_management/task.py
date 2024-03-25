
class Task:
    def __init__(self, id, name, description, status, value, dependencies):
        self.id = id
        self.name = name
        self.description = description
        self.status = status
        self.value = value
        self.dependencies = dependencies

    def update_status(self, new_status):
        self.status = new_status

    def add_dependency(self, task_id):
        self.dependencies.append(task_id)

    def remove_dependency(self, task_id):
        self.dependencies.remove(task_id)

    def to_json(self):
        json_task = {}
        for key, value in self.__dict__.items():
            json_task[key] = value
        return json_task
