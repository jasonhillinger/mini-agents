import importlib
import os
from dotenv import load_dotenv
from .LLMInterface import LLMInterface
import config
from LLM.MOCK_LLM import MOCK_LLM


class LLM:
    config: dict[str, dict[str, str]] = {}

    @staticmethod
    def loadConfig() -> None:
        load_dotenv()

        for key, value in os.environ.items():
            if "__" in key:
                section, subkey = key.split("__", 1)

                if section not in LLM.config:
                    LLM.config[section] = {}

                LLM.config[section][subkey] = value

    @staticmethod
    def factory(llmModel: str | None = None) -> LLMInterface:
        if config.TEST_MODE_ENABLED:
            return MOCK_LLM(
                apiKey="test_api_key",
                apiBaseUrl="http://localhost:11434",
                llmModel="MockLLM",
                maxRetries=10,
            )

        LLM.loadConfig()

        # For simplicity, we assume only one LLM model will be configured at a time.
        # We can extend this in the future to support multiple models if needed.
        if llmModel is None:
            selectedModel, modelConfig = next(iter(LLM.config.items()))
        else:
            selectedModel = llmModel
            if selectedModel not in LLM.config:
                raise ValueError(
                    f"LLM model '{selectedModel}' not found in configuration."
                )
            modelConfig = LLM.config[selectedModel]

        if not selectedModel:
            raise ValueError(
                "No LLM model selected. Set LLM_MODEL in .env or pass a model name to LLM.factory(...)."
            )

        apiBaseUrl = modelConfig.get("API_BASE_URL", "").strip()
        if not apiBaseUrl:
            raise ValueError(
                f"API_BASE_URL must be set for {selectedModel} in .env file."
            )

        llmModel = modelConfig.get("LLM_MODEL", "").strip()
        if not llmModel:
            raise ValueError(f"LLM_MODEL must be set for {selectedModel} in .env file.")

        # Not all models may require an API key, so we won't enforce it as strictly, but we'll still read it if it's there
        apiKey = (modelConfig.get("API_KEY") or "").strip()

        try:
            module = importlib.import_module(f"LLM.{selectedModel}")
            llm_class = getattr(module, selectedModel)
        except (ModuleNotFoundError, AttributeError) as exc:
            raise ValueError(f"Unsupported LLM type: {selectedModel}") from exc

        llm_instance = llm_class(
            apiKey=apiKey,
            apiBaseUrl=apiBaseUrl,
            llmModel=llmModel,
            maxRetries=int(modelConfig.get("MAX_RETRIES", 3)),
        )
        if not isinstance(llm_instance, LLMInterface):
            raise TypeError(f"{selectedModel} must implement LLMInterface.")

        return llm_instance
