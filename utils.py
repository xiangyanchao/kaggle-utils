import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np
import tensorflow as tf
import keras_core as keras

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

def get_devices():
    num_devices = len(jax.local_devices())
    print(f"Running on {num_devices} devices: {jax.local_devices()}")
    devices = mesh_utils.create_device_mesh((num_devices,))
    return devices

