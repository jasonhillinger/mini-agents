from typing import Literal, TypeAlias
from Utils.yamlUtils import getConfigFromYaml
from LLM.LLM import LLM
from Agents.AIAgent import AIAgent
import random
from termcolor import colored


# Example of debate; Two agents debate a topic for a specified amount of rounds.

Color: TypeAlias = Literal[
    "black", "grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
]


def run() -> None:
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
