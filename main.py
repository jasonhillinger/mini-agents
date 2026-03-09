import os
import importlib
from collections.abc import Callable
from pathlib import Path


def _loadOptions() -> dict[str, dict[str, Callable]]:
    options = {}
    choiceNum = 1
    BASE_DIR = Path(__file__).resolve().parent
    agentJobsDir = BASE_DIR / "AgentJobs"

    for directory in os.listdir(agentJobsDir):
        if not os.path.isdir(agentJobsDir / directory):
            continue

        for filename in os.listdir(agentJobsDir / directory):
            if filename == "runner.py":
                module = importlib.import_module(f"AgentJobs.{directory}.runner")
                options[str(choiceNum)] = {
                    "name": directory,
                    "func": getattr(module, "run"),
                }
                choiceNum += 1

    return options


def main() -> None:
    options = _loadOptions()
    print("Choose an option:")

    for key, func in options.items():
        print(f"{key}. {func.get('name')}")

    userChoice = input("Your choice: ").strip()

    if userChoice in options:
        options[userChoice]["func"]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
