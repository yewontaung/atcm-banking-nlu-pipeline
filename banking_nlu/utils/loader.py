import torch

def load_saved_model(model, saved_model_path:str, device:str):
    checkpoint = torch.load(
        saved_model_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model