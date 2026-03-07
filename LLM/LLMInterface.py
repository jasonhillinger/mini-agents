from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def __init__(
        self, apiKey: str, apiBaseUrl: str, llmModel: str, maxRetries: int
    ) -> None:
        self.API_KEY = apiKey
        self.API_BASE_URL = apiBaseUrl
        self.LLM_MODEL = llmModel
        self.MAX_RETRIES = maxRetries

    @abstractmethod
    def chatCompletion(self, messages: list[dict[str, str]]) -> str:
        return ""

    def getLlmModel(self) -> str:
        return self.LLM_MODEL

    def getApiKey(self) -> str:
        return self.API_KEY

    def getApiBaseUrl(self) -> str:
        return self.API_BASE_URL

    def getMaxAmountOfRetries(self) -> int:
        return self.MAX_RETRIES
