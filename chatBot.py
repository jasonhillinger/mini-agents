from AIAgent import AIAgent
from LLM import LLM

llm = LLM.factory()
jason = AIAgent(name="Jason", llm=llm, systemPrompt= "You are a backend software developer who expertise in PHP")


while True:
    message = input("> ").strip()
    response = jason.chat(message)
    print(response)
