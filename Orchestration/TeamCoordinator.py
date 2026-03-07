from Agents.AIAgent import AIAgent
import json


class TeamCoordinator:
    def __init__(
        self,
        agents: list[AIAgent],
        orchestrator: AIAgent
    ) -> None:
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
    
    def validateAgentPrompts(self, agentDict: dict) -> bool:
        """
        Validates that the dictionary has the correct structure:
        - Each value is a dict
        - Each dict contains 'systemPrompt' and 'userPrompt'
        - Both prompts are non-empty strings
        """
        if not isinstance(agentDict, dict):
            return False
        
        if (len(agentDict) == 0):
            return False

        for key, value in agentDict.items():
            # Each value must be a dict
            if not isinstance(value, dict):
                print(f"Value for '{key}' is not a dictionary")
                return False

            # Both required keys must exist
            for prompt_key in ["systemPrompt", "userPrompt"]:
                if prompt_key not in value:
                    print(f"Missing '{prompt_key}' in '{key}'")
                    return False

                # Each prompt must be a non-empty string
                if not isinstance(value[prompt_key], str) or not value[prompt_key].strip():
                    print(f"'{prompt_key}' in '{key}' must be a non-empty string")
                    return False

        return True

    def gatherAndThenAct(self) -> str:
        results = []
        for agent in self.agents:
            result = agent.chat()

            results.append(result)
        joined_results = "\n\n".join(results)

        orchTask = self.orchestrator.getPrompt() + (
            "Agent results:\n"
            f"{joined_results}\n"
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
                'The json structure must look like this:' 
                f'{{"agent0": {{"systemPrompt" : "insert your system prompt here #0", "userPrompt" : "insert your prompt here #0"}}, ... "agent{maxAmountOfWorkerAgents}": {{"systemPrompt" : "insert your system prompt here #{maxAmountOfWorkerAgents}", "userPrompt" : "insert your prompt here #{maxAmountOfWorkerAgents}"}}}}\n'
                'None of the systemPrompts or userPrompts should ever be empty'
        )
        self.orchestrator.appendSystemPrompt(systemPrompt)

        for attempt in range(maxAmountOfAttempts):
            orchestratorResult = self.orchestrator.chat()
            prompts            = self.loadJson(orchestratorResult)

            # Bad json, we try again
            if (not self.validateAgentPrompts(prompts)):
                print(f"Invalid JSON, attempting LLM request again...\nAttempt #{attempt + 1} out of {maxAmountOfAttempts}")
                continue
            
            # print(prompts)

            results = []
            for agentName, prompts in prompts.items():
                #TODO: make it possible to have agents with different LLMs
                agent = AIAgent(agentName, self.orchestrator.getLlm(), systemPrompt=prompts['systemPrompt'])
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
