from Orchestration.TeamCoordinator import TeamCoordinator
from LLM.LLM import LLM
from Agents.AIAgent import AIAgent
from Utils.yamlUtils import getConfigFromYaml


def initializeTeam(config) -> TeamCoordinator:
    llm = LLM.factory()

    # Create orchestrator object
    orchestrator = AIAgent(
        name="Orchestrator",
        llm=llm,
        systemPrompt=config["orchestrator"]["systemPrompt"],
        prompt=config["orchestrator"]["prompt"],
    )

    # Create agent objects
    agents = []
    for agentData in config["agents"]:
        agent = AIAgent(
            name=agentData["name"],
            llm=llm,
            systemPrompt=agentData["systemPrompt"],
            prompt=agentData["prompt"],
        )
        agents.append(agent)

    return TeamCoordinator(agents, orchestrator)


# Example of gather and then act; Many agents gather information,
# then the ochestrator agent takes that information and provides a final answer or solution to the problem at hand.
def run() -> None:
    config = getConfigFromYaml("gatherAndActAgents.yaml")

    team: TeamCoordinator = initializeTeam(config)
    finalResult = team.gatherAndThenAct()

    print(finalResult)
