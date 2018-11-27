import numpy as np
from .model import nn_model

class memory(list):
    def __init__(self,length):
        self.length = length
        super().__init__()
    def add(self,x):
        if len(self) < self.length:
            super().append(x)
        elif len(self) == self.length:
            super().pop(0)
            super().append(x)

def dqn(N):
    D = memory(N)
    Q = nn_model # implement two networks in one model with an update method.
    Qhat = Q.update()
    
