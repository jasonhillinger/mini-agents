# Mini-Agents

A lightweight Python framework for building and orchestrating LLM-powered AI agents. Create interactive multi-agent systems with easy YAML configuration and support for multiple LLM backends.

## Features

- **Easy Agent Creation** – Define agents with simple YAML configuration files
- **Multi-Agent Orchestration** – Coordinate multiple agents with the `TeamCoordinator`
- **Flexible LLM Support** – Pluggable LLM interface supporting GPT, LLAMA, and other models
- **Pre-built Agent Scenarios** – Ready-to-use agent configurations:
  - ChatBot – Interactive assistant agents
  - Debate – Multi-agent debate scenarios
  - Gather and Act – Collaborative information gathering
  - Lead and Direct – Hierarchical agent coordination
  - Reader Agents – Document and content analysis
- **RAG Integration** – Retrieval-Augmented Generation support
- **Agent Tools** – Extend agents with custom tools and capabilities

## Project Structure

```
mini-agents/
├── Agents/              # Core agent classes and tools
├── LLM/                 # LLM interface and implementations
├── AgentJobs/           # Pre-built agent scenarios
├── Orchestration/       # Multi-agent orchestration
├── Utils/               # Utilities (YAML, RAG, etc.)
├── tests/               # Unit tests
└── main.py              # Interactive launcher
```

## Quick Start

### Running the Interactive Menu

```bash
python3 main.py
```

This launches an interactive menu where you can choose from available agent scenarios:
- ChatBot
- Debate
- Gather and Act
- Lead and Direct
- Reader Agents
- Add your own!

### Running Unit Tests

```bash
python3 -m unittest discover -s tests
```

## Core Components

### AIAgent
The fundamental building block for creating an agent with a specific role and behavior.

```python
from Agents.AIAgent import AIAgent
from LLM.LLMInterface import LLMInterface

agent = AIAgent(
    name="Assistant",
    llm=llm_instance,
    systemPrompt="You are a helpful assistant",
    prompt="Initial prompt"
)
```

### TeamCoordinator
Orchestrates multiple agents to work together toward a common goal.

```python
coordinator = TeamCoordinator(agents=[agent1, agent2], orchestrator=main_agent)
```

### LLM Backends
Implement the `LLMInterface` to support different LLM providers:
- GPT-OSS 20B
- LLAMA 8B
- Custom implementations

### YAML Configuration
Define agent behavior in YAML files. Single agent example:

```yaml
agent:
  name: "PHP Expert"
  systemPrompt: "You are a backend software developer with expertise in PHP"
```

Multi-agent scenario:

```yaml
agents:
  - name: "Alice"
    systemPrompt: "You are a logical analyst"
    prompt: "Let's discuss..."
  - name: "Bob"
    systemPrompt: "You are a creative thinker"
    prompt: "I think differently..."
```

## Development

### Requirements
- Python 3.8+
- Dependencies managed via project configuration

### Adding New Agent Scenarios

1. Create a new directory in `AgentJobs/`
2. Add a `runner.py` file with a `run()` function
3. Add a `.yaml` config file for agent definitions
4. The scenario will automatically appear in the interactive menu

### API

To debugg, launch the server like this:
python3 -m debugpy --listen 5678 -m uvicorn api:app --reload

To lauch server normally:
uvicorn api:app --reload

In the file where run() exists, create an api() (TODO finish this documentation)

## Testing

The project includes comprehensive unit tests:
- `test_AIAgent.py` – Agent functionality tests
- `test_TeamCoordinator.py` – Orchestration tests

Run tests with:
```bash
python3 -m unittest discover -s tests
```

## License

See [LICENSE](LICENSE)
