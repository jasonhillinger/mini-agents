import unittest
from Agents.AIAgent import AIAgent
from LLM.LLM import LLM


class TestAIAgent(unittest.TestCase):
    def test_agentInitiation(self):
        llmMock = LLM.factory()  # You would replace this with a mock LLMInterface
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."
        prompt = "What is the capital of France?"

        agent = AIAgent(
            name=agentName,
            llm=llmMock,  # You would replace this with a mock LLMInterface
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
        llmMock = LLM.factory()  # You would replace this with a mock LLMInterface
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,  # You would replace this with a mock LLMInterface
            systemPrompt=systemPrompt,
        )

        userMessage = "How big is the the Earth?"
        agent.addMessage("user", userMessage)

        aiResponse = "The Earth is approximately 12,742 kilometers in diameter."
        agent.addMessage("assistant", aiResponse)
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
        llmMock = LLM.factory()  # You would replace this with a mock LLMInterface
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,  # You would replace this with a mock LLMInterface
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
        llmMock = LLM.factory()  # You would replace this with a mock LLMInterface
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,  # You would replace this with a mock LLMInterface
            systemPrompt=systemPrompt,
        )

        userMessage = "How big is the the Earth?"
        agent.addMessage("user", userMessage)

        aiResponse = "The Earth is approximately 12,742 kilometers in diameter."
        agent.addMessage("assistant", aiResponse)

        agent.reset()
        self.assertEqual(
            agent.getMessagesForChat(), [{"role": "system", "content": systemPrompt}]
        )

    def test_agentRevertPreviousConversation(self):
        llmMock = LLM.factory()  # You would replace this with a mock LLMInterface
        agentName = "TestAgent"
        systemPrompt = "You are a helpful assistant."

        agent = AIAgent(
            name=agentName,
            llm=llmMock,  # You would replace this with a mock LLMInterface
            systemPrompt=systemPrompt,
        )

        userMessage1 = "How big is the the Earth?"
        agent.addMessage("user", userMessage1)

        aiResponse1 = "The Earth is approximately 12,742 kilometers in diameter."
        agent.addMessage("assistant", aiResponse1)

        userMessage2 = "What about the Moon?"
        agent.addMessage("user", userMessage2)

        aiResponse2 = "The Moon is approximately 3,474 kilometers in diameter."
        agent.addMessage("assistant", aiResponse2)

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


if __name__ == "__main__":
    unittest.main()
