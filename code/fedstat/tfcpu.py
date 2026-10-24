"""Import this FIRST in every script: forces CPU-only TensorFlow and provides seeding."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU only (required by the run instructions)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Seed python, numpy and TensorFlow (as required for every run)."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


assert not tf.config.list_physical_devices("GPU"), "GPU visible - this project is CPU-only"
