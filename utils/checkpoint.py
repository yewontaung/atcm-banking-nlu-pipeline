import torch


def save_checkpoint(
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