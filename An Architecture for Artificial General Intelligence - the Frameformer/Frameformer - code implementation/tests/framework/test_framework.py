import unittest
from unittest.mock import patch
from AutoGPT.source.framework.framework import Framework


class TestFramework(unittest.TestCase):

    def setUp(self):
        self.framework = Framework()

    # @patch('AutoGPT.source.framework.framework.Framework.test_program_routine')
    def test_test_program_routine(self):
        # Setup mock return value
        # mock_execute.return_value = ("Hello World", "")

        # Create a Framework instance and call test_program_routine
        output, error = self.framework.execute_program_routine("print('Hello World')")

        # Asserts
        self.assertEqual(output, "Hello World")
        self.assertEqual(error, "")

    # ... other test cases for test_program_routine

    # @patch('AutoGPT.source.framework.framework.Framework.determine_optimal_framework_type')
    def test_determine_optimal_framework_type(self):
        # Setup mock OpenAI response
        # mock_openai.return_value = "text"

        # Create a Framework instance and call determine_optimal_framework_type
        text_framework = self.framework.determine_optimal_framework_type(objective="summarize a given sentence")
        program_framework = self.framework.determine_optimal_framework_type(
            objective="predict the next chess move given a "
                      "chess board")

        print(text_framework, program_framework)
        # Assert
        self.assertEqual(text_framework, "text")
        self.assertEqual(program_framework, "program")
        self.assertEqual(self.framework.id, 1)
        self.assertEqual(self.framework.name, "")

    # ... other test cases for determine_optimal_framework_type

    def test_update_information(self):
        self.framework.update_information(name="test", description="test", text_routine="test", program_routine="test",
                                          program_instructions="test")
        self.assertEqual(self.framework.name, "test")
        self.assertEqual(self.framework.objective, "test")
        self.assertEqual(self.framework.text_routine, "test")
        self.assertEqual(self.framework.program_routine, "test")
        self.assertEqual(self.framework.program_instructions, "test")


if __name__ == '__main__':
    unittest.main()
