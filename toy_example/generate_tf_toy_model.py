import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils import wrap_frozen_graph

import sys
import numpy as np
import os


class add_two_inputs(tf.keras.Model):

    @tf.function
    def __call__(self, x):
        a, b = x
        z = tf.math.add(a, b, name='ADD')
        return z



def train():

    # Initialize the defined tf model
    model = add_two_inputs()
   
    full_model = tf.function(lambda x: model(x))
    # Specify the inputs, shape and type, for the tf model now wrapped in a tf function. In this case we have just two integers.
    full_model = full_model.get_concrete_function(
    x=[tf.TensorSpec(shape=(1,), dtype=tf.dtypes.int32), tf.TensorSpec(shape=(1,), dtype=tf.dtypes.int32)])
    
    # This function is only present in TF >= 2.0v
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()
    
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="toy_model.pb",
                      as_text=False)

    # Here you can load your already frozen and test if you get the expected outputs.
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/toy_model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    # We haven't specified the input/output node names, so we get the default values.
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
            inputs=["x:0", "x_1:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    
    # Let's define some inputs values
    a = 3
    b = 4

    # Get predictions from frozen model
    c = frozen_func(x=tf.constant(a), x_1=tf.constant(b))
    print('{} + {} = {}'.format(a, b, c[0]))

if __name__ == '__main__':
    train()
