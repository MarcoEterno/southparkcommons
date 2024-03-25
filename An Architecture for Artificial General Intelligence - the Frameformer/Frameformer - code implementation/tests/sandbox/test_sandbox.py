import unittest
import subprocess
from unittest.mock import patch, MagicMock
import os
from AutoGPT.source.sandbox.sandbox import execute_program_routine_python


class TestExecuteProgramRoutinePython(unittest.TestCase):

    def setUp(self):
        self.script_name = "tester_program.py"
        self.script_path = os.path.join(os.getcwd(), 'programs', self.script_name)


    def test_execute_successful(self):
        # Directly execute the script and test for a successful outcome
        output, error = execute_program_routine_python(self.script_name)
        # Assuming the script prints "Success" on successful execution
        self.assertEqual(output, "Hello World!\n")
        self.assertEqual(error, "")

    @patch('subprocess.run')
    def test_execute_successful(self, mock_run):
        # Mock subprocess.run to simulate successful execution
        mock_run.return_value = subprocess.CompletedProcess(args=['python3', self.script_path], returncode=0,
                                                            stdout="Success", stderr="")

        output, error = execute_program_routine_python(self.script_name)
        self.assertEqual(output, "Success")
        self.assertEqual(error, "")
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_execute_with_error(self, mock_run):
        # Mock subprocess.run to simulate execution with error
        mock_run.return_value = subprocess.CompletedProcess(args=['python3', self.script_path], returncode=1, stdout="",
                                                            stderr="Error")

        output, error = execute_program_routine_python(self.script_name)
        self.assertEqual(output, "")
        self.assertEqual(error, "Error")
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_execute_timeout(self, mock_run):
        # Mock subprocess.run to raise a TimeoutExpired exception
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['python3', self.script_path], timeout=30)

        with self.assertRaises(subprocess.TimeoutExpired):
            execute_program_routine_python(self.script_name)

    @patch('subprocess.run')
    def test_unexpected_exception(self, mock_run):
        # Mock subprocess.run to raise an unexpected exception
        mock_run.side_effect = Exception("Unexpected error")

        with self.assertRaises(Exception) as context:
            execute_program_routine_python(self.script_name)
        self.assertTrue('Unexpected error' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
