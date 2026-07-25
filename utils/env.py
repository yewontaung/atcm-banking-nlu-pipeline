import os
from dotenv import load_dotenv


load_dotenv()

TRAINING_FILE = os.getenv("TRAINING_FILE")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")