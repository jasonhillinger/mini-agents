import json
import urllib.error
import urllib.request
from .LLMInterface import LLMInterface


class GPT_OSS_20B(LLMInterface):
    def __init__(
        self, apiKey: str, apiBaseUrl: str, llmModel: str, maxRetries: int = 3
    ):
        super().__init__(apiKey, apiBaseUrl, llmModel, maxRetries)

    def chatCompletion(self, messages: list[dict[str, str]]) -> str:
        return self._executePostRequest("/v1/chat/completions", messages)

    def _executePostRequest(self, endpoint: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.getLlmModel(),
            "messages": messages,
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
                f"Could not reach VKS LLM. Ensure {self.getLlmModel()} is installed and you are connected to the VPN."
            ) from exc

        return data["choices"][0]["message"]["content"]
