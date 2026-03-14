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
            "You are an orchestrator agent whose job is to ask agent(s) to read files and answer questions about the files based on their content. "
            "You will be given a list of file paths. You should ask one or more agents to read the files and then answer questions about the files based on their content. "
            "Your response should be in JSON format with the following structure:\n {{'filePath': 'path/to/file', 'question': 'question about the file'}, {'filePath': 'path/to/file', 'question': 'question about the file'}...}\n"
        ),
    )

    team = TeamCoordinator([], orchestrator=orchestrator)
    team.readerJobs(directory)
