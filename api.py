from fastapi import FastAPI
import os
import importlib
from collections.abc import Callable
from pathlib import Path
import re
import inspect

app = FastAPI()

AGENT_JOB_PREFIX = "/run-agent-job"


def _loadAPIOptions() -> dict[str, dict[str, Callable]]:
    options = {}

    BASE_DIR = Path(__file__).resolve().parent
    agentJobsDir = BASE_DIR / "AgentJobs"

    for directory in os.listdir(agentJobsDir):
        if not os.path.isdir(agentJobsDir / directory):
            continue

        for filename in os.listdir(agentJobsDir / directory):
            if filename == "runner.py":
                module = importlib.import_module(f"AgentJobs.{directory}.runner")

                if not hasattr(module, "apiGet") and not hasattr(module, "apiPost"):
                    continue

                options[directory] = {
                    "name": directory,
                    "getFunc": getattr(module, "apiGet", None),
                    "postFunc": getattr(module, "apiPost", None),
                }

    return options


def formatEndpoint(name: str) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower().strip()
    return f"{AGENT_JOB_PREFIX}/{name}"


options = _loadAPIOptions()
availableRoutes = {}


def create_endpoint(func: Callable):
    async def endpoint_func(*args, **kwargs):
        return func(*args, **kwargs)

    setattr(endpoint_func, "__signature__", inspect.signature(func))

    return endpoint_func


for name, option in options.items():
    getFunc = option["getFunc"]
    postFunc = option["postFunc"]
    endpoint = formatEndpoint(name)

    if getFunc:
        getName = f"{name}|GET"
        app.add_api_route(
            endpoint,
            create_endpoint(getFunc),
            methods=["GET"],
            name=getName,
        )
        availableRoutes[getName] = endpoint

    if postFunc:
        postName = f"{name}|POST"
        app.add_api_route(
            endpoint,
            create_endpoint(postFunc),
            methods=["POST"],
            name=postName,
        )
        availableRoutes[postName] = endpoint


@app.get("/available-agent-jobs")
def available_agent_jobs():
    return availableRoutes
