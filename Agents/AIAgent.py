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
        self.systemPrompt = systemPrompt
        self.messages = []

    def getPrompt(self) -> str:
        return self.prompt

    def getLlm(self) -> LLMInterface:
        return self.llm

    def getName(self) -> str:
        return self.name

    def reset(self) -> None:
        self.messages = [self.messages[0]]

    def getSystemPromptStructured(self) -> dict:
        return {"role": "system", "content": self.systemPrompt}

    def getSystemPrompt(self) -> str:
        return self.systemPrompt

    def appendSystemPrompt(self, prompt: str) -> None:
        self.systemPrompt = self.systemPrompt + "\n" + prompt

    def updateSystemPrompt(self, prompt: str) -> None:
        self.systemPrompt = prompt

    def getMessagesForChat(self) -> list:
        return [self.getSystemPromptStructured(), *self.messages]

    def addMessage(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def chat(self, userMessage: str | None = None) -> str:
        if userMessage is None:
            userMessage = self.getPrompt()

        self.addMessage("user", userMessage)
        messages = self.getMessagesForChat()
        aiReply = self.getLlm().chatCompletion(messages)

        self.addMessage("assistant", aiReply)
        return aiReply
