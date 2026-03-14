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


if __name__ == "__main__":
    unittest.main()
