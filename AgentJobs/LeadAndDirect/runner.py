from Utils.yamlUtils import getConfigFromYaml
from LLM.LLM import LLM
from Agents.AIAgent import AIAgent
from Orchestration.TeamCoordinator import TeamCoordinator


# Example of lead and direct; The orchestrator agent leads the team by giving each agent specific tasks to do,
# then gathers the results and provides a final answer or solution to the problem at hand.
def run() -> None:
    config = getConfigFromYaml("leadAndDirectAgents.yaml")

    llm = LLM.factory()

    # Create orchestrator object
    orchestrator = AIAgent(
        name="Orchestrator",
        llm=llm,
        systemPrompt=config["orchestrator"]["systemPrompt"],
        prompt=config["orchestrator"]["prompt"],
    )

    maxAmountOfAgents = config["orchestrator"]["maxAmountOfAgents"]
    team = TeamCoordinator([], orchestrator=orchestrator)
    finalResult = team.leadAndDirectAgents(maxAmountOfAgents)

    print(finalResult)
