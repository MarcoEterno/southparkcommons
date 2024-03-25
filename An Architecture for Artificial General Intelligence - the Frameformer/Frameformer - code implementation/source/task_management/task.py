
class Task:
    def __init__(self, id, name, description, status, value, dependencies):
        self.id = id
        self.name = name
        self.description = description
        self.status = status
        self.value = value
        self.dependencies = dependencies
        self.

    def update_status(self, new_status):
        self.status = new_status

    def add_dependency(self, task_id):
        self.dependencies.append(task_id)

    def remove_dependency(self, task_id):
        self.dependencies.remove(task_id)

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "value": self.value,
            "dependencies": self.dependencies
        }
