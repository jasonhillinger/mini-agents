from Agents.AIAgent import AIAgent
from LLM.LLM import LLM
from Utils.yamlUtils import getConfigFromYaml


def run() -> None:
    config = getConfigFromYaml("chatBot.yaml")
    name = config["agent"]["name"]
    systemPrompt = config["agent"]["systemPrompt"]

    llm = LLM.factory()
    agent = AIAgent(
        name=name,
        llm=llm,
        systemPrompt=systemPrompt,
    )

    while True:
        message = input("You > ").strip()
        response = agent.chat(message)
        print(f"{agent.getName()} > {response}")
