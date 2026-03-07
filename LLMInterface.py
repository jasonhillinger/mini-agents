from abc import ABC, abstractmethod

class LLMInterface(ABC):

    @abstractmethod
    def __init__(self, api_key: str, api_base_url: str, llm_model: str, max_retries: int) -> None:
        self.API_KEY = api_key
        self.API_BASE_URL = api_base_url
        self.LLM_MODEL = llm_model
        self.MAX_RETRIES = max_retries

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


