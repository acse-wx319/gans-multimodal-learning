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

#################################
# General functions
#################################

def weights_init(m):
    """
    Initializes the weights of Generator/Disriminator.

    Parameters
    ----------
    m : 
        An object of the nn.Module class.

    Returns
    -------
    None.

    """
    if type(m) == torch.nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def plot_data(true_data, fake_data, out_name, out_path, plot_3d=False):
    """
    Plots generated data and real data and saves plots to local folder. 

    Parameters
    ----------
    true_data : numpy array
        True data.
    fake_data : numpy array
        Generated data.
    out_name : string
        Output file name suffix.
    out_path : string 
        Path to save the output plot.
    plot_3d : boolean, optional
        Whether plotting in 3d. The default is False.

    Returns
    -------
    None.

    """       
    with torch.no_grad():

        plt.clf()
        
        if not plot_3d:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            plt.scatter(true_data[:, 0], true_data[:, 1], c='orange', label='Real data')
            ax.legend()
            plt.savefig(out_path + 'real' + out_name + '.jpg')
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            plt.scatter(fake_data[:, 0], fake_data[:, 1], c='green', label='Generated data')
            ax.legend()
            plt.savefig(out_path + 'fake' + out_name + '.jpg')
        else:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], c='orange', label='Real data')
            ax.legend()
            plt.savefig(out_path + 'real' + out_name + '.jpg')
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(fake_data[:, 0], fake_data[:, 1], fake_data[:, 2], c='green', label='Generated data')
            ax.legend()
            plt.savefig(out_path + 'fake' + out_name + '.jpg')


def make_data_iterator(dataset, batch_size):
    """
    Returns an iterator that always generates a new random sample for the specified distribution.

    Parameters
    ----------
    dataset : string
        Name of the synthetic dataset to generate.
    batch_size : int
        Mini batch size.

    Yields
    ------
    numpy array
        A sample from the specified dataset.

    """
    
    if dataset == '8gaussians':
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2) * .2
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = torch.Tensor(dataset)
            dataset /= 1.414  # stdev
            yield dataset
            
    elif dataset == 'sine':
        while True:
            noise = 0.2
            x = torch.linspace(-4, 4, batch_size, dtype=torch.float32)
            y = np.sin(x) + noise*np.random.randn(*x.shape)
            yield torch.stack([x, y], dim=1)
            
    elif dataset == 'heteroscedastic':
        theta = torch.linspace(0, 2, batch_size)
        x = np.exp(theta)*np.tan(0.1*theta)
        while True:
            b = (0.001 + 0.5 * np.abs(x)) * np.random.normal(1, 1, batch_size)
            y = np.exp(theta)*np.sin(0.1*theta) + b
            yield torch.stack([x, y], dim=1)
            
    elif dataset == 'moon':
        noise = 0.1
        while True:
            data, _ = sklearn.datasets.make_moons(n_samples=batch_size,
                                                  noise=noise)
            yield torch.Tensor(data)
    
    elif dataset == 'helix':
        noise = 0.2
        while True:
            t = torch.linspace(0, 20, batch_size)
            x = np.cos(t)
            x2 = np.sin(t) + noise * np.random.randn(*x.shape)
    
            yield torch.stack([x, x2, t], dim=1)
    
    elif dataset == 'circle':
        while True:
            t = np.random.random(batch_size) * 2 * np.pi - np.pi
            length = 1 - np.random.random(batch_size)*0.4
            x = torch.Tensor(np.multiply(np.cos(t), length))
            y = torch.Tensor(np.multiply(np.sin(t), length))
    
            yield torch.stack([x, y], dim=1)

    elif dataset == '2spirals':
        while True:
            z = torch.randn(batch_size, 2)
            n = torch.sqrt(torch.rand(batch_size // 2)) * 540 * (2 * math.pi) / 360
            d1x = - torch.cos(n) * n + torch.rand(batch_size // 2) * 0.5
            d1y =   torch.sin(n) * n + torch.rand(batch_size // 2) * 0.5
            x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                           torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
            yield x + 0.1*z


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, lbd, use_cuda = False):
    """
    Calculates the gradient penalty term for WGAN

    Parameters
    ----------
    netD : Disriminator object
        Disriminator.
    real_data : torch.Tensor
        Read data.
    fake_data : torch.Tensor
        Data genereated by GAN.

    Returns
    -------
    gradient_penalty : torch.Tensor
        The gradient penalty term for WGAN.

    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if use_cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()

    disc_interpolates = netD(interpolates)


    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lbd

    return gradient_penalty
    

def predict_fixed(netG, fixed_x, input_size, output_size, input_dim, latent_dim, use_cuda=False):
    """
    For some given x coordinates and a pre-trained generator, 
    run an optimization algorithm to find latent vectors that 
    would generate data with these x coordinates (as close as possible). 
    For each x, generate input_size number of datasets and 
    select output_size number of datasets that have the closest x values
    to fixed_x. 
        

    Parameters
    ----------
    netG : Generator
        a pre-trained generator object.
    fixed_x : float
        a value for x.
    input_size : int
        how many values to generate. must be greater than output_size.
    output_size : int
        how many predictions to make.
    input_dim : int
        input dimension of the generator 
    latent_dim : int
        latent dimension of the generator 
    use_cuda : boolean
        whether using GPU

    Returns
    -------
    The fixed-input predictions.

    """
    # placeholder
    true_dist = torch.zeros((input_size, input_dim), dtype=torch.float32)
    # fix x
    true_dist[:, 0] = torch.Tensor([fixed_x]*input_size)
    
    # mse loss
    mse = nn.MSELoss()
    noise = torch.randn(input_size, latent_dim)
    noise.requires_grad = True
    
    if use_cuda:
        noise = noise.cuda()
        true_dist = true_dist.cuda()
    
    # use two learning rates for better performance
    optimizerL = optim.Adam([noise], lr=1e-4)
    optimizerL1 = optim.Adam([noise], lr=5e-5)
    losses = []
    min_loss = None
    
    # run first optimizer
    for epoch in range(10000):
        samples = netG(noise)
        loss = mse(samples[:, 0], true_dist[:, 0])
        if not min_loss or loss.item() < min_loss:
            min_loss = loss.item() 
            min_latent = noise.detach()
        losses.append(loss.item())
        loss.backward()
        optimizerL.step()
        
    # run second optimizer
    for epoch in range(30000):
        samples = netG(noise)
        loss = mse(samples[:, 0], true_dist[:, 0])
        if not min_loss or loss.item() < min_loss:
            min_loss = loss.item() 
            min_latent = noise.detach()
        losses.append(loss.item())
        loss.backward()
        optimizerL1.step()
    
    # create samples using the optimimal latent space
    samples = netG(min_latent).detach()
    xs = samples[:, 0]
    x = true_dist[0, 0].item()
    # sort by absolute value of the difference between 
    # generated x and fixed_x
    diff = abs(xs - x)  
    indices = torch.argsort(diff)
    indices_sub = indices[:output_size]
    
    return samples[indices_sub, :]



