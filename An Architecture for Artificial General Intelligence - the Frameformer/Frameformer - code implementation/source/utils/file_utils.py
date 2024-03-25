
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return None

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)
