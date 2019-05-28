# -*- coding: utf-8 -*-
from dival import LearnedReconstructor
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D


class LearnedPrimalDualReconstructor(LearnedReconstructor):
    def __init__(self):
        self.model = Input(shape=(None, None, 1))
        
    def reconstruct(self, observation_data):
        sess = tf.InteractiveSession()

    def train(self, dataset, forward_op=None):
        pass
