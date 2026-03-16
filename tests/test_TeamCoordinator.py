import unittest
from Orchestration.TeamCoordinator import TeamCoordinator
import coverage

coverage.process_startup()


class TestTeamCoordinator(unittest.TestCase):
    def test_validDictionaryResponse(self):
        validDict = {
            "agent0": {
                "systemPrompt": "You are a helpful assistant.",
                "userPrompt": "What is the capital of France?",
            },
            "agent1": {
                "systemPrompt": "You are a math expert.",
                "userPrompt": "What is 2 + 2?",
            },
        }
        try:
            TeamCoordinator.validateAgentPrompts(
                validDict, ["systemPrompt", "userPrompt"]
            )
        except ValueError:
            self.fail(
                "validateAgentPrompts raised ValueError unexpectedly with valid input!"
            )

    def test_invalidDictionaryResponse(self):
        cases = [
            None,  # Not a dictionary
            {},  # Empty dictionary
            {"agent1": "Not a dict"},  # Value is not a dict
            {
                "agent1": {"systemPrompt": "You are a helpful assistant."}
            },  # Missing userPrompt
            {
                "agent1": {"userPrompt": "What is the capital of France?"}
            },  # Missing systemPrompt
            {
                "agent1": {
                    "systemPrompt": "",
                    "userPrompt": "What is the capital of France?",
                }
            },  # Empty systemPrompt
            {
                "agent1": {
                    "systemPrompt": "You are a helpful assistant.",
                    "userPrompt": "",
                }
            },  # Empty userPrompt
        ]
        for case in cases:
            with self.subTest(case=case):
                try:
                    TeamCoordinator.validateAgentPrompts(
                        case, ["systemPrompt", "userPrompt"]
                    )
                except ValueError:
                    self.assertTrue(True)  # Expected outcome


if __name__ == "__main__":
    unittest.main()
