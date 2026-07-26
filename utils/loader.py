from utils import env


def load_modelname():
    return f"{env.SAVED_MODEL_NAME_PREFIX}_d{env.TRAINING_DATASIZE}_e{env.EPOCHS}.pt"