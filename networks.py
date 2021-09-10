import os
import threading
import numpy as np

class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass

def set_session(sess): pass

graph = DummyGraph()
sess = None

if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense , LSTM , Conv2D , BatchNormalization, Dropout, MaxPooling2D , Flatten
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.backend import set_session
    import tensorflow as tf
    graph = tf.get_default_graph()
    sess = tf.compat.v1.Session()
else:
    print('Nooooo~~')

class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, shared_network = None, activation = 'sigmoid' , loss = 'mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    def predict(self,sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

    def train_on_batch(self,x, y):
        loss = 0.
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                loss = self.model.train_on_batch(x, y)
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
            