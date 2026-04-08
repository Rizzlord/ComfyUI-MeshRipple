# config_loader.py
import yaml
import argparse

class DictToObject:
    """Recursively convert dictionaries to objects for dot-notation access."""
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f'{self.__dict__}'

def load_config():
    """
    Parse command-line arguments to get the config file path,
    load the YAML file, and return a configuration object.
    """
    parser = argparse.ArgumentParser(description="Load training configuration from a YAML file.")
    parser.add_argument('--config', type=str, default='./config_loader/config.yaml',
                        help='Path to the configuration YAML file.')
    
    args = parser.parse_args()
    print(f"config path :{args.config}")

    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)

    return DictToObject(config_dict)
