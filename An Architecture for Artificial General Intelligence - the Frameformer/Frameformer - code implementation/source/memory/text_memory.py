
class TextMemory:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_memory(self):
        try:
            with open(self.file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return "Memory file not found."

    def write_memory(self, content):
        with open(self.file_path, 'w') as file:
            file.write(content)

    def update_memory(self, content):
        current_memory = self.read_memory()
        updated_memory = current_memory + "\n" + content
        self.write_memory(updated_memory)

    def delete_memory(self):
        with open(self.file_path, 'w') as file:
            file.write("")  # Clear the file content
