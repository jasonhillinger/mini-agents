from Agents.AIAgent import AIAgent
from LLM.LLMInterface import LLMInterface
from Agents.AgentTools import AgentTools


class AgentTooler(AIAgent):
    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        systemPrompt: str = "You are a helpful AI assistant.",
        prompt: str = "",
        tools: dict = {},
    ) -> None:
        super().__init__(name=name, llm=llm, systemPrompt=systemPrompt, prompt=prompt)
        self.tools = tools

    def readFile(self, filePath: str) -> None:
        if not self.tools.get("readFile", False):
            raise Exception("Tool not available")

        content = AgentTools.readFile(filePath)
        self.setExtraSystemPrompt(
            f"The content of the file {filePath} is:\n{content.getContent()}"
        )

    def readFileContent(self, content: str):
        if not self.tools.get("readFile", False):
            raise Exception("Tool not available")

        self.setExtraSystemPrompt(f"The content of the file is:\n{content}")
