from pathlib import Path
import inspect
import yaml


def getConfigFromYaml(yamlFilename: str) -> dict:
    # gets the config from the yaml file in the same directory as the runner.py file that calls this function
    frame = inspect.stack()[1]
    BASE_DIR = Path(frame.filename).resolve().parent
    path = BASE_DIR / yamlFilename
    with open(path) as file:
        config = yaml.safe_load(file)

    return config
