from Orchestration.TeamCoordinator import TeamCoordinator
from Agents.AIAgent import AIAgent
from LLM.LLM import LLM
import yaml
import random
from termcolor import colored
from typing import Literal, TypeAlias
from pathlib import Path

Color: TypeAlias = Literal[
    "black", "grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
]


def getConfigFromYaml(yamlFilename: str) -> dict:
    BASE_DIR = Path(__file__).resolve().parent
    path = BASE_DIR / yamlFilename
    with open(path) as file:
        config = yaml.safe_load(file)

    return config


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
def gatherAndThenAct() -> None:
    config = getConfigFromYaml("gatherAndActAgents.yaml")

    team: TeamCoordinator = initializeTeam(config)
    finalResult = team.gatherAndThenAct()

    print(finalResult)


# Example of lead and direct; The orchestrator agent leads the team by giving each agent specific tasks to do,
# then gathers the results and provides a final answer or solution to the problem at hand.
def leadAndDirect() -> None:
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


# Example of debate; Two agents debate a topic for a specified amount of rounds.
def debate() -> None:
    amountOfRounds = int(input("How many rounds should this debate go for? "))
    config = getConfigFromYaml("debate.yaml")

    llm = LLM.factory()

    agents = []
    for agentData in config["agents"]:
        agent = AIAgent(
            name=agentData["name"],
            llm=llm,
            systemPrompt=agentData["systemPrompt"],
            prompt=agentData["prompt"],
        )
        agents.append(agent)

    if len(agents) != 2:
        print(
            "Debate currently only supports 2 debaters per debate! Adjust the yaml file to have 2 agents."
        )
        return

    # Randomly choose who starts
    currentAgent: AIAgent = random.choice(agents)

    otherAgent: AIAgent = agents[0] if currentAgent == agents[1] else agents[1]

    currentColor = 0

    colors: list[tuple[Color, Color]] = [("red", "magenta"), ("cyan", "green")]

    print("\n--- Debate Start ---\n")

    currentPrompt = otherAgent.getPrompt()
    print(
        colored(f"{otherAgent.getName()}", colors[currentColor][0])
        + ": "
        + colored(f"{currentPrompt}", colors[currentColor][1])
    )
    currentColor = 1
    for roundNum in range(amountOfRounds):
        print(f"\n=== Round {roundNum + 1} ===")

        # Current agent responds
        response = currentAgent.chat(currentPrompt)
        print(
            colored(f"{currentAgent.getName()}", colors[currentColor][0])
            + ": "
            + colored(f"{response}", colors[currentColor][1])
        )
        currentColor = 1 if currentColor == 0 else 0
        # Next agent uses this response as their prompt
        currentPrompt = response

        # Swap turns
        currentAgent, otherAgent = otherAgent, currentAgent

    print("\n--- Debate End ---")


choices = {
    "1": gatherAndThenAct,
    "2": leadAndDirect,
    "3": debate,
}


def main() -> None:
    print("Choose an option:")

    for key, func in choices.items():
        print(f"{key}. {func.__name__}")

    userChoice = input("Your choice: ")

    choices.get(userChoice, lambda: print("Invalid choice"))()


if __name__ == "__main__":
    main()
