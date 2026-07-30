import torch

from banking_nlu.utils import env


def load_modelname():
    return f"{env.SAVED_MODEL_NAME_PREFIX}_d{env.TRAINING_DATASIZE}_e{env.EPOCHS}.pt"

def load_saved_model(model, saved_model_path:str, device:str):
    checkpoint = torch.load(
        saved_model_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model