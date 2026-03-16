import unittest
from Agents.AIAgent import AIAgent
from LLM.MOCK_LLM import MOCK_LLM


class TestAIAgent(unittest.TestCase):
    @staticmethod
    def getMockLLM() -> MOCK_LLM:
        return MOCK_LLM("abc123", "http://localhost:11434", "MockLLM", 10)

    def test_agentInitiation(self):
        llmMock = self.getMockLLM()
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."
        prompt = "What is the capital of France?"

        agent = AIAgent(
            name=agentName,
            llm=llmMock,
            systemPrompt=systemPrompt,
            prompt=prompt,
        )
        self.assertEqual(agent.getName(), agentName)
        self.assertEqual(agent.getSystemPrompt(), systemPrompt)
        self.assertEqual(
            agent.getSystemPromptStructured(),
            {"role": "system", "content": systemPrompt},
        )
        self.assertEqual(agent.getPrompt(), prompt)

    def test_agentMockChat(self):
        llmMock = self.getMockLLM()
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,
            systemPrompt=systemPrompt,
        )

        userMessage = MOCK_LLM.CHAT_MESSAGES[0]
        aiResponse = agent.chat(userMessage)

        messagesForChat = agent.getMessagesForChat()
        self.assertEqual(
            messagesForChat,
            [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userMessage},
                {"role": "assistant", "content": aiResponse},
            ],
        )

    def test_agentGetMessagesForChat(self):
        llmMock = self.getMockLLM()
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,
            systemPrompt=systemPrompt,
        )

        systemPromptDict = {"role": "system", "content": systemPrompt}
        self.assertEqual(agent.getMessagesForChat(), [systemPromptDict])

        userMessage = {"role": "user", "content": "Hello!"}
        agent.addMessage(userMessage["role"], userMessage["content"])
        self.assertEqual(
            agent.getMessagesForChat(),
            [
                systemPromptDict,
                userMessage,
            ],
        )

        assistantMessage = {
            "role": "assistant",
            "content": "Hi there! How can I assist you today?",
        }
        agent.addMessage(assistantMessage["role"], assistantMessage["content"])
        self.assertEqual(
            agent.getMessagesForChat(),
            [
                systemPromptDict,
                userMessage,
                assistantMessage,
            ],
        )

    def test_agentReset(self):
        llmMock = self.getMockLLM()
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,
            systemPrompt=systemPrompt,
        )

        userMessage = MOCK_LLM.CHAT_MESSAGES[1]
        agent.chat(userMessage)

        agent.reset()
        self.assertEqual(
            agent.getMessagesForChat(), [{"role": "system", "content": systemPrompt}]
        )

    def test_agentRevertPreviousConversation(self):
        llmMock = self.getMockLLM()
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,
            systemPrompt=systemPrompt,
        )

        userMessage1 = MOCK_LLM.CHAT_MESSAGES[0]
        aiResponse1 = agent.chat(userMessage1)

        userMessage2 = MOCK_LLM.CHAT_MESSAGES[1]
        aiResponse2 = agent.chat(userMessage2)

        beforeRevertMessages = agent.getMessagesForChat()
        self.assertEqual(
            beforeRevertMessages,
            [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userMessage1},
                {"role": "assistant", "content": aiResponse1},
                {"role": "user", "content": userMessage2},
                {"role": "assistant", "content": aiResponse2},
            ],
        )

        agent.revertPreviousConversation()

        # After reverting, the last user and assistant messages should be removed, leaving only the system prompt and the first user and assistant messages.
        self.assertEqual(
            agent.getMessagesForChat(),
            [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userMessage1},
                {"role": "assistant", "content": aiResponse1},
            ],
        )

    def test_agentRevertPreviousConversationWithNoMessages(self):
        llmMock = self.getMockLLM()
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,
            systemPrompt=systemPrompt,
        )

        agent.revertPreviousConversation()

        # After reverting, the last user and assistant messages should be removed, leaving only the system prompt and the first user and assistant messages.
        self.assertEqual(
            agent.getMessagesForChat(),
            [
                {"role": "system", "content": systemPrompt},
            ],
        )


if __name__ == "__main__":
    unittest.main()
