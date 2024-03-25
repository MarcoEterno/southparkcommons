import unittest
from AutoGPT.source.framework.framework import Framework

class TestFramework(unittest.TestCase):

    def setUp(self):
        self.framework = Framework()

    def test_determine_optimal_framework_type(self):
        objective = "predicting the next chess move given a chess board"
        result = self.framework.determine_optimal_framework_type(objective)
        self.assertIn(result, ["text", "program"])

    def test_create_new_framework(self):
        name = "Test Framework"
        description = "Test Description"
        self.framework.create_new_framework(name=name, objective=description)
        self.assertEqual(self.framework.name, name)
        self.assertEqual(self.framework.objective, description)

    def test_update_information(self):
        self.framework.update_information(name="Updated Name")
        self.assertEqual(self.framework.name, "Updated Name")

    def test_add_information(self):
        self.framework.add_information(knowledge_of_task="Test Knowledge")
        self.assertIn("Test Knowledge", self.framework.knowledge_of_task)

    def test_remove_information(self):
        self.framework.add_information(knowledge_of_task="Test Knowledge")
        self.framework.remove_information(knowledge_of_task="Test Knowledge")
        self.assertNotIn("Test Knowledge", self.framework.knowledge_of_task)

    def test_get_information(self):
        self.framework.add_information(knowledge_of_task="Test Knowledge")
        info = self.framework.get_information("knowledge_of_task")
        self.assertEqual(info, {"knowledge_of_task": ["Test Knowledge"]})

    def test_to_json(self):
        json_framework = self.framework.to_json()
        self.assertIsInstance(json_framework, dict)

if __name__ == "__main__":
    unittest.main()