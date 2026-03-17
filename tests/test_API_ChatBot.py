import unittest

from fastapi.testclient import TestClient
from LLM.MOCK_LLM import MOCK_LLM
from api import app
import config
import coverage

coverage.process_startup()

client = TestClient(app)


class TestAPIChatBot(unittest.TestCase):
    def setUp(self):
        config.TEST_MODE_ENABLED = True

    def test_chatBotEndpoint(self):
        message = MOCK_LLM.CHAT_MESSAGES[0]
        payload = {"messages": [{"role": "user", "content": message}]}

        response = client.post("/run-agent-job/chat-bot", json=payload)

        self.assertEqual(response.status_code, 200)
        responseData = response.json()
        aiResponse = responseData[-1]["content"]
        expectedResponse = MOCK_LLM.ASSISTANT_RESPONSES[0]
        self.assertEqual(aiResponse, expectedResponse)

    def test_invalidRequestChatBotEndpoint(self):
        payload = {"messages": "This should cause an error"}

        response = client.post("/run-agent-job/chat-bot", json=payload)

        self.assertEqual(response.status_code, 422)
        responseData = response.json()
        self.assertEqual(
            responseData["detail"][0]["msg"], "Input should be a valid list"
        )

    def test_emptyMessagesList(self):
        payload = {"messages": []}

        response = client.post("/run-agent-job/chat-bot", json=payload)

        self.assertEqual(response.status_code, 400)
        responseData = response.json()
        self.assertEqual(responseData["detail"], "Missing 'messages' field")

    def test_getSystemPrompt(self):
        response = client.get("/run-agent-job/chat-bot")

        self.assertEqual(response.status_code, 200)
        responseData = response.json()
        self.assertIsInstance(responseData, str)


if __name__ == "__main__":
    unittest.main()
