import subprocess
import shlex
from typing import Tuple
import os


def execute_program_routine_python(script_name: str) -> Tuple[str, str]:
    """
    Executes the program routine of the framework.

    Args:
        script_name: Name of the Python script to execute.

    Returns:
        Tuple[str, str]: Output and error of the program routine.
    """
    # Prepare the command to run the Python script
    # script_path = f"AutoGPT/sandbox/programs/{script_name}"
    script_path = os.path.join(os.path.join(os.getcwd(), 'programs'), script_name)
    command = f"python3 {script_path}"

    # Use shlex to safely split the command string
    args = shlex.split(command)

    try:
        # Execute the script in a new process
        # stdout and stderr capture the output and errors
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

        # Check if the script ran successfully
        if result.returncode == 0:
            print("Script executed successfully.")
            print("Output:\n", result.stdout)
        else:
            print("Script execution failed.")
            print("Error:\n", result.stderr)

        return result.stdout, result.stderr

    except subprocess.TimeoutExpired as e:
        print("Script execution exceeded the time limit.")
        raise e

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
