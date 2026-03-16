from .LLMInterface import LLMInterface


class MOCK_LLM(LLMInterface):
    CHAT_MESSAGES = ["Hey there! How are you?", "How big is the the Earth?"]

    ASSISTANT_RESPONSES = [
        "Hello! I'm doing well, thank you. How can I assist you today?",
        "The Earth is approximately 12,742 kilometers in diameter.",
    ]

    def __init__(
        self, apiKey: str, apiBaseUrl: str, llmModel: str, maxRetries: int = 3
    ):
        super().__init__(
            apiKey=apiKey,
            apiBaseUrl=apiBaseUrl,
            llmModel=llmModel,
            maxRetries=maxRetries,
        )
        if len(self.CHAT_MESSAGES) != len(self.ASSISTANT_RESPONSES):
            raise ValueError(
                "CHAT_MESSAGES and ASSISTANT_RESPONSES must have the same length"
            )

        self.MOCK_CONVERSATION = dict(zip(self.CHAT_MESSAGES, self.ASSISTANT_RESPONSES))

    def chatCompletion(self, messages: list[dict[str, str]]) -> str:
        message = messages[-1]["content"]
        if message not in self.MOCK_CONVERSATION:
            raise ValueError(f"Message '{message}' not found in MOCK_CONVERSATION")

        return self.MOCK_CONVERSATION[message]
