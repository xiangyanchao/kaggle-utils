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

# Replicate the model and optimizer variable on all devices
def get_replicated_train_state(devices,model,optimizer):
    # All variables will be replicated on all devices
    var_mesh = Mesh(devices, axis_names=("_"))
    # In NamedSharding, axes not mentioned are replicated (all axes here)
    var_replication = NamedSharding(var_mesh, PartitionSpec())

    # Apply the distribution settings to the model variables
    trainable_variables     = jax.device_put(model.trainable_variables,     var_replication)
    non_trainable_variables = jax.device_put(model.non_trainable_variables, var_replication)
    optimizer_variables     = jax.device_put(optimizer.variables,           var_replication)

    # Combine all state in a tuple
    return (trainable_variables, non_trainable_variables, optimizer_variables)

_model=None
_loss_fn=None
_optimizer=None
_compute_gradients=None
def set_model_suit(model,loss_fn,optimizer):
    global _model,_loss_fn,_optimizer,_compute_gradients
    _model=model
    _loss_fn=loss_fn
    _optimizer=optimizer
    _compute_gradients=jax.value_and_grad(compute_loss, has_aux=True)

# This is the loss function that will be differentiated.
# Keras provides a pure functional forward pass: model.stateless_call
def compute_loss(trainable_variables, non_trainable_variables, x, y):
    y_pred, updated_non_trainable_variables = _model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss_value = _loss_fn(y, y_pred)
    return loss_value, updated_non_trainable_variables

# Training step, Keras provides a pure functional optimizer.stateless_apply
@jax.jit
def train_step(train_state, x, y):
    trainable_variables, non_trainable_variables, optimizer_variables = train_state
    (loss_value, non_trainable_variables), grads = _compute_gradients(trainable_variables, non_trainable_variables, x, y)
    trainable_variables, optimizer_variables = _optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
    return loss_value, (trainable_variables,non_trainable_variables,optimizer_variables)

def get_data_sharding(devices,test_shard_data=None):
    data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
    data_sharding = NamedSharding(data_mesh,PartitionSpec("batch",),)  # naming axes of the sharded partition
    if test_shard_data is not None:
        # Display data sharding
        print("Data sharding...")
        test_shard_data = jax.device_put(test_shard_data.numpy(), data_sharding)
        jax.debug.visualize_array_sharding(test_shard_data)
        print("Data sharding over!")
    return data_sharding
