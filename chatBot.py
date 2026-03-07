from Agents.AIAgent import AIAgent
from LLM.LLM import LLM

llm = LLM.factory()
agent = AIAgent(
    name="PHP Expert",
    llm=llm,
    systemPrompt="You are a backend software developer who expertise in PHP",
)

while True:
    message = input("You > ").strip()
    response = agent.chat(message)
    print(f"{agent.getName()} > {response}")
