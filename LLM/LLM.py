import importlib
import os
from dotenv import load_dotenv
from .LLMInterface import LLMInterface

class LLM:

    NO_LLM_MODEL_CHOSEN = "NO_LLM_MODEL_CHOSEN"
    REQUIRED_API_KEY_ENV_VAR = "API_KEY"

    config = {}

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
    def factory(llm_model: str | None = None) -> LLMInterface:
        LLM.loadConfig()

        # For simplicity, we assume only one LLM model will be configured at a time. 
        # We can extend this in the future to support multiple models if needed. 
        if llm_model is None:
            selected_model, model_config = next(iter(LLM.config.items()))
        else:
            selected_model = llm_model
            if selected_model not in LLM.config:
                raise ValueError(f"LLM model '{selected_model}' not found in configuration.")
            model_config = LLM.config[selected_model]

        if not selected_model:
            raise ValueError(
                "No LLM model selected. Set LLM_MODEL in .env or pass a model name to LLM.factory(...)."
            )
        
        api_base_url = model_config.get('API_BASE_URL', "").strip()
        if not api_base_url:
            raise ValueError(f"API_BASE_URL must be set for {selected_model} in .env file.")
        
        llm_model = model_config.get('LLM_MODEL', "").strip()
        if not llm_model:
            raise ValueError(f"LLM_MODEL must be set for {selected_model} in .env file.")

        # Not all models may require an API key, so we won't enforce it as strictly, but we'll still read it if it's there
        api_key = (model_config.get('API_KEY') or "").strip()

        try:
            module = importlib.import_module(f"LLM.{selected_model}")
            llm_class = getattr(module, selected_model)
        except (ModuleNotFoundError, AttributeError) as exc:
            raise ValueError(f"Unsupported LLM type: {selected_model}") from exc

        llm_instance = llm_class(api_key=api_key, api_base_url=api_base_url, llm_model=llm_model, max_retries=int(model_config.get('MAX_RETRIES', 3)))
        if not isinstance(llm_instance, LLMInterface):
            raise TypeError(f"{selected_model} must implement LLMInterface.")

        return llm_instance
