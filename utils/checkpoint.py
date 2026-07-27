import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    metadata=None
):

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    if metadata:
        checkpoint.update(metadata)


    torch.save(
        checkpoint,
        path
    )