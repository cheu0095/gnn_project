import yaml

def load_config(config_file: str) -> dict:
    with open("config\\" + config_file + ".yaml", "r") as file:
        config = yaml.safe_load(file)
    return config