from .LLMInterface import LLMInterface
import urllib.error
import urllib.request
import json

class LLAMA_8B_llm(LLMInterface):

    def __init__(self, api_key: str, api_base_url: str, llm_model: str, max_retries: int = 3):
        super().__init__(api_key=api_key, api_base_url=api_base_url, llm_model=llm_model, max_retries=max_retries)
    
    def chatCompletion(self, messages: list[dict[str, str]]) -> str:
        return self._executePostRequest("/api/chat", messages)
    
    def _executePostRequest(self, endpoint: str, messages: list[dict[str, str]]) -> str: 
        payload = {
            "model": self.getLlmModel(),
            "messages": messages,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.getApiKey():
            headers["Authorization"] = f"Bearer {self.getApiKey()}"

        request = urllib.request.Request(
            url=f"{self.getApiBaseUrl()}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach the LLM. Please ensure {self.getLlmModel()} is installed and that the required API key is configured. A VPN may also be required.") from exc
        
        return data["message"]["content"]
