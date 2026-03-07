from LLM.LLMInterface import LLMInterface


class AIAgent:
    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        systemPrompt: str = "You are a helpful AI assistant.",
        prompt: str = "",
    ) -> None:
        self.name = name
        self.llm = llm
        self.prompt = prompt
        self.messages = [{"role": "system", "content": systemPrompt}]

    def getPrompt(self) -> str:
        return self.prompt

    def getLlm(self) -> LLMInterface:
        return self.llm

    def getName(self) -> str:
        return self.name

    def reset(self) -> None:
        self.messages = [self.messages[0]]

    def appendSystemPrompt(self, prompt: str) -> None:
        for message in self.messages:
            if message["role"] == "system":
                message["content"] += "\n" + prompt
                return

    def updateSystemPrompt(self, prompt: str) -> None:
        for message in self.messages:
            if message["role"] == "system":
                message["content"] = prompt
                return

    def chat(self, userMessage: str | None = None) -> str:
        if userMessage is None:
            userMessage = self.getPrompt()

        self.messages.append({"role": "user", "content": userMessage})
        aiReply = self.getLlm().chatCompletion(self.messages)

        self.messages.append({"role": "assistant", "content": aiReply})
        return aiReply
