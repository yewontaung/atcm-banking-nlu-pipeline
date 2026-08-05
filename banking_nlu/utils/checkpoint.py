import torch


def save_model(
    path,
    model,
):

    saved = {
        "model": model.state_dict(),
    }

    torch.save(
        saved,
        path
    )