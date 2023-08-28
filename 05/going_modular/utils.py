"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch as pt
from pathlib import Path
from sys import path
def save_model(
    model:pt.nn.Module,
    target_dir:str,
    model_name:str):
    """Saves a PyTorch model to target dir"""
    # Create dirs
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name should end with '.pt or .pth"
    model_save_path = target_dir_path/model_name

    print(f'[INFO] Saving model to: {model_save_path}')
    pt.save(obj=model.state_dict(), f=model_save_path)

def path_dir_with_level(level:int=0,path_chosen:str=path) -> str:
    """This is a function which returns the current directory from a level of deepness
    Denna funktionen ger nuvarande foldern eller tidigare foldrar med ett visst djup"""
    return "/".join(map(str,(path_chosen[0].split("\\")[:-level]))) if level != 0 else path[0]