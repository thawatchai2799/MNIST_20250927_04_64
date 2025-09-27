#--- 2025-09-27 02-48 – by Dr. Thawatchai Chomsiri
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Model
import datetime

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Flatten, Dense

import datetime
import numpy as np
import os
import re
import math
import pickle

#def lsgelu(x):    # Left-Shifted GELU with 1 range
#    return x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0)))

def lsgelus300(x):    
    S=3.00
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus270(x):    
    S=2.70
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus240(x):    
    S=2.40
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus210(x):    
    S=2.10
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus180(x):    
    S=1.80
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus150(x):    # LSGELU   
    S=1.50
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus120(x):    
    S=1.20
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus090(x):    
    S=0.90
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus060(x):    
    S=0.60
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus030(x):    
    S=0.30
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelus000(x):    # GELU    
    S=0.00
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))


def lsgelu9999(x):    
    S=3.71901648546
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9950(x):    
    S=2.57582930355
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9900(x):    
    S=2.32634787404
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9750(x):    
    S=1.95996398454
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9500(x):    
    S=1.64485362695
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

#-----
def lsgelu9332(x): # LSGELU   
    S=1.5
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

#-----
def lsgelu9250(x):    
    S=1.43953147094
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu9000(x):    
    S=1.28155156554
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu8000(x):    
    S=0.841621233573
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu7500(x):    
    S=0.674489750196
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

def lsgelu6666(x):    
    S=0.430727299295
    return x * 0.5 * (1 + tf.math.erf((x + S) / tf.sqrt(2.0)))

#-----
def lsgelu2(x):    # Left-Shifted GELU with 2 range
    return tf.where(
        x >= 0,
        x, 
        x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0)))
    ) 

def lsgelu3(x):    # Left-Shifted GELU with 3 range
    L = -3.00
    return tf.where(
        x >= 0,
        x,
        tf.where(
            x >= L,
            x * 0.5 * (1 + tf.math.erf((x + 1.5) / tf.sqrt(2.0))),
            tf.zeros_like(x)
        )
    ) 

#def build_model(activation_fn):
#    inputs = tf.keras.Input(shape=(28, 28, 1))
#    x = Conv2D(128, (3, 3), activation=activation_fn)(inputs)
#    x = MaxPooling2D((2, 2))(x)
#    x = Flatten()(x)    
#    x = Dense(128, activation=activation_fn)(x)
#    x = Dense(128, activation=activation_fn)(x)    
#    outputs = Dense(10, activation='softmax')(x)
#    model = tf.keras.Model(inputs, outputs)
#    return model

def build_model(activation_fn):
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs) 
    x = tf.keras.layers.Dense(128, activation=activation_fn)(x)    
    x = tf.keras.layers.Dense(128, activation=activation_fn)(x)    
    x = tf.keras.layers.Dense(128, activation=activation_fn)(x)    
    x = tf.keras.layers.Dense(128, activation=activation_fn)(x)    
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# แก้ไขส่วนโหลดข้อมูล
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ขนาดข้อมูล
img_rows, img_cols = 28, 28

# ปรับ shape ของข้อมูลเพื่อให้ตรงกับ Input ของโมเดล (28, 28, 1)
x_train = x_train.reshape(-1, img_rows, img_cols, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, img_rows, img_cols, 1).astype('float32') / 255.0

activations_list = {
#    "LSGELUS300": lsgelus300,
#    "LSGELUS270": lsgelus270,
#    "LSGELUS240": lsgelus240,
#    "LSGELUS210": lsgelus210,
#    "LSGELUS180": lsgelus180,
#    "LSGELUS150": lsgelus150,  # LSGELU
#    "LSGELUS120": lsgelus120,
#    "LSGELUS090": lsgelus090,
#    "LSGELUS060": lsgelus060,
#    "LSGELUS030": lsgelus030,  
#    'GELU': tf.nn.gelu,
#    'ELU': tf.nn.elu,
#    'ReLU': tf.nn.relu,
#    'Swish': tf.nn.swish,     


    'GELU': tf.nn.gelu,
    "LSGELUS150": lsgelus150,  # LSGELU
    "LSGELUS180": lsgelus180,
    "LSGELUS120": lsgelus120,
    'ELU': tf.nn.elu,
    'ReLU': tf.nn.relu,
    'Swish': tf.nn.swish,     
    "LSGELUS210": lsgelus210,
    "LSGELUS090": lsgelus090,
    "LSGELUS240": lsgelus240,
    "LSGELUS060": lsgelus060,
    "LSGELUS270": lsgelus270,

    "LSGELUS300": lsgelus300,  
    "LSGELUS030": lsgelus030,

    
}

epochs = 101  ################
num_runs = 1 ###############
batch_size = 64 ###############
results = {
    'activation': [],
    'accuracy_per_epoch': []
}
accuracy_summary = {}

for run_idx in range(num_runs):
    print(f"\n--- Run {run_idx:03d} of {num_runs:03d} ---")
    results = {}
    accuracy_results = {}
    loss_results = {}
    
    for act_name, act_fn in activations_list.items():
        print(f"Running: Activation={act_name} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        model = build_model(act_fn)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
        
        print(f"Running: Activation={act_name} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")   
        results[act_name] = {key: np.array(val) for key, val in history.history.items()}
        print(f"Details in results for round {run_idx:03d}:")
        for act_name, metrics_dict in results.items():
            print(f"\nActivation: {act_name}")
            for metric_name, metric_values in metrics_dict.items():
                print(f"  {metric_name}: {metric_values}")

        np.savez(f"accuracy_{run_idx:03d}_{act_name}.npz", accuracy=np.array(history.history['accuracy']))
        np.savez(f"loss_{run_idx:03d}_{act_name}.npz", loss=np.array(history.history['loss']))
        
        accuracy_results[act_name] = np.array(history.history['accuracy'])
        loss_results[act_name] = np.array(history.history['loss'])

    print(f" ----- Data -------- ")
#    results[act_name] = {key: np.array(val) for key, val in history.history.items()}
#    print(f"Details in results for round {run_idx:03d}:")
#    for act_name, metrics_dict in results.items():
#        print(f"\nActivation: {act_name}")
#        for metric_name, metric_values in metrics_dict.items():
#            print(f"  {metric_name}: {metric_values}")

    print(f" ------------------- ")


   
print(f"\nEND at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
