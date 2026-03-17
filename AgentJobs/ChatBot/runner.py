from Agents.AIAgent import AIAgent
from LLM.LLM import LLM
from Utils.yamlUtils import getConfigFromYaml
from pydantic import BaseModel
from fastapi import HTTPException


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel):
    messages: list[Message]


def initializeAgent() -> AIAgent:
    config = getConfigFromYaml("chatBot.yaml")
    name = config["agent"]["name"]
    systemPrompt = config["agent"]["systemPrompt"]

    llm = LLM.factory()
    return AIAgent(
        name=name,
        llm=llm,
        systemPrompt=systemPrompt,
    )


def apiPost(data: Request):
    if len(data.messages) == 0:
        raise HTTPException(status_code=400, detail="Missing 'messages' field")

    message = data.messages[-1].content
    agent = initializeAgent()
    agent.chat(message)
    return agent.getMessagesForChat()


def apiGet():
    agent = initializeAgent()
    return agent.getSystemPrompt()


def run() -> None:
    agent = initializeAgent()

    while True:
        message = input("You > ").strip()
        response = agent.chat(message)
        print(f"{agent.getName()} > {response}")
