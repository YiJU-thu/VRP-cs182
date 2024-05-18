import sys
from utils_project.utils_func import cstring as C

def check_module(module_name, install_cmd=""):
    try:
        __import__(module_name)
        print(f"> {C.blue(module_name)} is installed.")
    except ImportError:
        print(f"* {C.red(module_name)} is not installed. Install by:\n {install_cmd}")



if __name__ == "__main__":

    # Example usage
    required_modules = {
        "torch": ...,
        "torchvision": ...,
        "tensorflow": ...,
        "tensorboard_logger": "pip install tensorboard_logger",
        "tqdm": ...,
        "pickle": ...,
        "loguru": "pip install loguru",
        "wandb": "pip install wandb\n(and add your API key to environment variable (user settings -> danger zone -> API keys): export WANDB_API_KEY=<your_api_key_here>)",
        "concorde": "See: https://github.com/jvkersch/pyconcorde?tab=readme-ov-file#how-do-i-install-it",
    }

    for module, install_cmd in required_modules.items():
        check_module(module, install_cmd)