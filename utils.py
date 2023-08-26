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

_loss_fn=None
def set_loss_fn(loss_fn):
    global _loss_fn
    _loss_fn=loss_fn

_model=None
def set_model(model):
    global _model
    _model=model

# This is the loss function that will be differentiated.
# Keras provides a pure functional forward pass: model.stateless_call
def compute_loss(trainable_variables, non_trainable_variables, x, y):
    y_pred, updated_non_trainable_variables = _model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss_value = _loss_fn(y, y_pred)
    return loss_value, updated_non_trainable_variables

_compute_gradients=None
def set_compute_gradients():
    global _compute_gradients
    _compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)

# Training step, Keras provides a pure functional optimizer.stateless_apply
@jax.jit
def train_step(train_state, x, y):
    trainable_variables, non_trainable_variables, optimizer_variables = train_state
    (loss_value, non_trainable_variables), grads = _compute_gradients(trainable_variables, non_trainable_variables, x, y)
    trainable_variables, optimizer_variables = optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
    return loss_value, (trainable_variables,non_trainable_variables,optimizer_variables)

def visualize_array_sharding(devices,train_data):
    data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
    data_sharding = NamedSharding(data_mesh,PartitionSpec("batch",),)  # naming axes of the sharded partition
    
    # Display data sharding
    x, y = next(iter(train_data))
    print('x:',x.shape)
    sharded_x = jax.device_put(x.numpy(), data_sharding)
    sharded_x = jax.numpy.reshape(sharded_x, [sharded_x.shape[0], -1])
    print('sharded_x:',sharded_x.shape)
    print("Data sharding...")
    jax.debug.visualize_array_sharding(sharded_x)
    print("Data sharding over.")

