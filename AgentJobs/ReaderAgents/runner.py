from Utils.yamlUtils import getConfigFromYaml
from LLM.LLM import LLM
from Agents.AIAgent import AIAgent
from Orchestration.TeamCoordinator import TeamCoordinator
from pathlib import Path


def run() -> None:
    config = getConfigFromYaml("readingAgents.yaml")

    llm = LLM.factory()

    directory = Path(config["directory"])

    # Create orchestrator object
    orchestrator = AIAgent(
        name="Orchestrator",
        llm=llm,
        systemPrompt=(
            "You are an orchestrator agent whose job is to format the responses from the reader agents into a final answer to the user's question. "
            "You should use the responses from the reader agents to create a final answer to the user's question. "
            "You should throw out any irrelevant information from the reader agents and only include information that is relevant to the user's question. "
            "If the reader agents' responses contradict each other, you should use your best judgement to determine which response is more likely to be correct based on the content of the responses. "
            "If the readers agents is indicating that they are not able to answer the question, you should discard their response and not include it in the final answer. "
        ),
    )

    team = TeamCoordinator([], orchestrator=orchestrator)
    team.readerJobs(directory)
