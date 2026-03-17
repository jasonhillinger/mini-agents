import unittest

from fastapi.testclient import TestClient
from api import app
import config
import coverage

coverage.process_startup()

client = TestClient(app)


class TestAPIChatBot(unittest.TestCase):
    def setUp(self):
        config.TEST_MODE_ENABLED = True

    def test_getSystemPrompt(self):
        response = client.get("/available-agent-jobs")

        self.assertEqual(response.status_code, 200)
        responseData = response.json()
        self.assertIsInstance(responseData, dict)


if __name__ == "__main__":
    unittest.main()
