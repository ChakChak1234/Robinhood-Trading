import pandas as pd
import numpy as np
import random

class MLP:
    def __init__(self,data,inputs,targets):
        self.data = data
        self.inputs = inputs.to_numpy()
        self.targets = targets.to_numpy()
        
        self.input_size = len(self.inputs[0])
        self.num_inputs = len(inputs)
        
        self.weights = [random.random() for _ in range(self.input_size + 1)]
    
    def activation(self,n):
        if n > 0:
            return 1
        return 0
    
    def train(self):
        for j in range(10):
            for i in range(self.num_inputs):
                y = self.predict(self.inputs[i])
                e = self.targets[i] - y
                #wx = self.dot(self.inputs[i],self.weights)
                
                if e == 0:
                    continue
                elif e > 0:
                    self.weights = self.add(self.weights,self.inputs[i])
                else:
                    self.weights = self.sub(self.weights,self.inputs[i])
        return self.weights
    
    def predict(self,inputs):
        z = self.dot(self.weights,inputs)
        a = self.activation(z)
        return a
    
    """
    Return dot product of two vectors
    """
    def dot(self, a, b):
        x = 0
        for i,j in zip(a,b):
            x += i*j
        return x

    """
    Return addition of two vectors
    """
    def add(self, a, b):
        x = []
        for i,j in zip(a,b):
            x.append(i+j)
        return x

    """
    Return subtraction of two vectors
    """
    def sub(self, a, b):
        x = []
        for i,j in zip(a,b):
            x.append(i-j)
        return x
    
    