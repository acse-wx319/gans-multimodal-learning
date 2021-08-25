import random
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    """
    Defines a generator object for GAN.
    """
    def __init__(self, LATENT_DIM, DIM, DROPOUT_RATE, OUTPUT_DIM):

        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(LATENT_DIM, DIM),
            nn.Dropout(p=DROPOUT_RATE),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.Dropout(p=DROPOUT_RATE),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.Dropout(p=DROPOUT_RATE),
            nn.ReLU(True),
            nn.Linear(DIM, OUTPUT_DIM)
        )
        
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class DiscriminatorSN(nn.Module):
    """
    Defines a discriminator object for GAN, with spectral normalization.
    """
    def __init__(self, DIM, INPUT_DIM):
        super(DiscriminatorSN, self).__init__()
        
        main = nn.Sequential(
            spectral_norm(nn.Linear(INPUT_DIM, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, DIM)),
            nn.ReLU(True),
            nn.Linear(DIM, 1)
        )
            
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)

class Discriminator(nn.Module):
    """
    Defines a discriminator object for GAN, without spectral normalization.
    """
    def __init__(self, DIM, INPUT_DIM):
        super(Discriminator, self).__init__()
        
        main = nn.Sequential(
            nn.Linear(INPUT_DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1)
        )
            
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)

class DiscriminatorSNTime(nn.Module):
    """
    Defines a discriminator object for GAN, with spectral normalization.
    """
    def __init__(self, DIM, INPUT_DIM, BATCH_SIZE):
        super(DiscriminatorSNTime, self).__init__()
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        main = nn.Sequential(
            spectral_norm(nn.Linear(INPUT_DIM, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, DIM)),
            nn.ReLU(True),
            nn.Linear(DIM, 1)
        )
            
        self.main = main

    def forward(self, inputs):
        inputs = inputs.view(self.batch_size, self.input_dim)
        output = self.main(inputs)
        return output.view(-1)

class DiscriminatorTime(nn.Module):
    """
    Defines a discriminator object for GAN, without spectral normalization.
    """
    def __init__(self, DIM, INPUT_DIM, BATCH_SIZE):
        super(DiscriminatorTime, self).__init__()
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        main = nn.Sequential(
            nn.Linear(INPUT_DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1)
        )
            
        self.main = main

    def forward(self, inputs):
        inputs = inputs.view(self.batch_size, self.input_dim)
        output = self.main(inputs)
        return output.view(-1)