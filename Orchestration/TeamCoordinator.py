from pathlib import Path

from Agents.AIAgent import AIAgent
from Agents.AgentTooler import AgentTooler
import json


class TeamCoordinator:
    def __init__(self, agents: list[AIAgent], orchestrator: AIAgent) -> None:
        self.agents = agents
        self.orchestrator = orchestrator

    def addAgent(self, agent: AIAgent) -> None:
        self.agents.append(agent)

    def loadJson(self, jsonString: str) -> dict:
        try:
            return json.loads(jsonString)  # Try parsing the string
        except json.JSONDecodeError:
            print("An error occured while decoding JSON.")
            return {}

    def validateAgentPrompts(self, agentDict: dict, requiredKeys: list) -> bool:
        """
        Validates that the dictionary has the correct structure:
        - Each value is a dict
        - Each dict contains 'systemPrompt' and 'userPrompt'
        - Both prompts are non-empty strings
        """
        if not isinstance(agentDict, dict):
            return False

        if len(agentDict) == 0:
            return False

        for key, value in agentDict.items():
            # Each value must be a dict
            if not isinstance(value, dict):
                print(f"Value for '{key}' is not a dictionary")
                return False

            # Both required keys must exist
            for promptKey in requiredKeys:
                if promptKey not in value:
                    print(f"Missing '{promptKey}' in '{key}'")
                    return False

                # Each prompt must be a non-empty string
                if (
                    not isinstance(value[promptKey], str)
                    or not value[promptKey].strip()
                ):
                    print(f"'{promptKey}' in '{key}' must be a non-empty string")
                    return False

        return True

    def gatherAndThenAct(self) -> str:
        results = []
        for agent in self.agents:
            result = agent.chat()

            results.append(result)
        joinedResults = "\n\n".join(results)

        orchTask = self.orchestrator.getPrompt() + (
            "Agent results:\n" f"{joinedResults}\n"
        )

        return self.orchestrator.chat(orchTask)

    def leadAndDirectAgents(self, maxAmountOfWorkerAgents: int) -> str:
        maxAmountOfAttempts = self.orchestrator.getLlm().getMaxAmountOfRetries()
        systemPrompt = (
            "You will create prompts for your AI agents.\n"
            "Your prompts must be directive so that the agents do your requested task as accurately as possible.\n"
            f"You have {maxAmountOfWorkerAgents} agent(s) to work with.\n"
            "Never direct your agents to direct each other. Only you should direct them.\n"
            "Your output must be in valid JSON format where each element is a JSON which contains a system prompt and user prompt for the agents.\n"
            "The json structure must look like this:"
            f'{{"agent0": {{"systemPrompt" : "insert your system prompt here #0", "userPrompt" : "insert your prompt here #0"}}, ... "agent{maxAmountOfWorkerAgents}": {{"systemPrompt" : "insert your system prompt here #{maxAmountOfWorkerAgents}", "userPrompt" : "insert your prompt here #{maxAmountOfWorkerAgents}"}}}}\n'
            "None of the systemPrompts or userPrompts should ever be empty"
        )
        self.orchestrator.appendSystemPrompt(systemPrompt)

        for attempt in range(maxAmountOfAttempts):
            orchestratorResult = self.orchestrator.chat()
            prompts = self.loadJson(orchestratorResult)

            # Bad json, we try again
            if not self.validateAgentPrompts(prompts, ["systemPrompt", "userPrompt"]):
                print(
                    f"Invalid JSON, attempting LLM request again...\nAttempt #{attempt + 1} out of {maxAmountOfAttempts}"
                )
                continue

            # print(prompts)

            results = []
            for agentName, prompts in prompts.items():
                # TODO: make it possible to have agents with different LLMs
                agent = AIAgent(
                    agentName,
                    self.orchestrator.getLlm(),
                    systemPrompt=prompts["systemPrompt"],
                )
                # self.addAgent(agent)

                result = f"{agent.getName()} : {agent.chat(prompts['userPrompt'])}"
                print("\033[31m" + result + "\033[0m")
                results.append(result)

            joinedResults = "\n\n".join(results)

            systemPrompt = (
                "You created prompts for your AI agents.\n"
                "Your prompts were be directive so that the agents do your requested task as accurately as possible.\n"
                "You did this to help you achieve your initial goal.\n"
                f"Here is the output of your agents: {joinedResults}"
            )
            self.orchestrator.updateSystemPrompt(systemPrompt)

            return self.orchestrator.chat()

        return "LLM failed to format response correctly."

    def readerJobs(self) -> None:
        readerAgents = {}
        while True:
            userQuestion = input("> ").strip()
            if userQuestion.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            for attempt in range(self.orchestrator.getLlm().getMaxAmountOfRetries()):
                orchestratorResult = self.orchestrator.chat(userQuestion)
                results = self.loadJson(orchestratorResult)

                # Bad json, we try again
                if not self.validateAgentPrompts(results, ["filePath", "question"]):
                    print(
                        f"Invalid JSON, attempting LLM request again...\nAttempt #{attempt + 1} out of {self.orchestrator.getLlm().getMaxAmountOfRetries()}"
                    )
                    continue

                for value in results.values():
                    filePath = value["filePath"]
                    question = value["question"]

                    lastModified = Path(filePath).stat().st_mtime

                    if filePath not in readerAgents:
                        readerAgent = AgentTooler(
                            name=f"ReaderAgent_{filePath}",
                            llm=self.orchestrator.getLlm(),
                            tools={"readFile": True},
                            systemPrompt="You are an expert of the file you are reading. Answer questions about the file based on its content.",
                        )

                        readerAgent.readFile(filePath)

                        readerAgents[filePath] = {
                            "agent": readerAgent,
                            "lastModified": lastModified,
                            "question": question,
                            "response": None,
                        }
                    elif lastModified > readerAgents[filePath]["lastModified"]:
                        readerAgents[filePath]["agent"].readFile(filePath)
                        readerAgents[filePath]["lastModified"] = lastModified

                    readerAgents[filePath]["response"] = readerAgents[filePath][
                        "agent"
                    ].chat(question)
                    readerAgents[filePath]["question"] = question

                self.orchestrator.updateSystemPrompt(
                    "You created questions for your reader agents based on the files they read."
                )

                print(
                    self.orchestrator.chat(
                        "Here are the questions and their corresponding file paths:\n"
                        + "\n".join(
                            [
                                f"{filePath} got question: {value['question']} with response: {value['response']}"
                                for filePath, value in readerAgents.items()
                            ]
                        )
                    )
                )
