import os, sys

sys.path.append(os.getcwd())

import random

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import math
from mpl_toolkits.mplot3d import Axes3D

import tflib as lib
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import timeit

torch.manual_seed(1)

# ==================Specifications Start======================

MODE = 'wgan-gp'  # wgan or wgan-gp
DATASET = 'sine'  # 8gaussians, sine, heteroscedastic, moon
suffix = '_dropout'
DIM = 512  # 512 Model dimensionality
LATENT_DIM = 2 # latent space dimension
INPUT_DIM = 2 # input dimension
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 256  # 256 Batch size
ITERS = 10000  # 100000, how many generator iterations to train for
use_cuda = False
plot_density = False
plot_contour = False
plot_3d = (DATASET == 'helix')


TMP_PATH = 'tmp/' + DATASET + suffix + '/'
if not os.path.isdir(TMP_PATH):
    os.makedirs(TMP_PATH)

if DATASET == 'sine':
    RANGE = 4
elif DATASET == 'heteroscedastic':
    RANGE = 1.5
elif DATASET == '8gaussians':
    RANGE = 3
elif DATASET == 'moon':
    RANGE = 2
else:
    RANGE = 3

# ==================Specifications End======================

# ==================Definition Start======================

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(LATENT_DIM, DIM),
            nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(DIM, INPUT_DIM),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(INPUT_DIM, DIM),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            # nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

frame_index = [0]
def generate_image(true_dist, out_name, plot_contour=True, plot_density=False,
                   plot_3d=False):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """    
    with torch.no_grad():
        noise = torch.randn(BATCH_SIZE, LATENT_DIM)
        if use_cuda:
            noise = noise.cuda()
            true_dist = true_dist.cuda()
        samples = netG(noise, true_dist).cpu().data.numpy()
        
        # contour plot
        plt.clf()
        if plot_contour:
            N_POINTS = 128
            points = torch.zeros((N_POINTS, N_POINTS, 2), dtype=torch.float32)
            points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
            points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
            points = points.reshape((-1, 2))
            if use_cuda:
                points = points.cuda()
            disc_map = netD(points).cpu().data.numpy()
            x = y = np.linspace(-RANGE, RANGE, N_POINTS)
            plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
        
        if not plot_3d:
            plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange')
            if not FIXED_GENERATOR:
                plt.scatter(samples[:, 0], samples[:, 1], c='green')
            plt.savefig(TMP_PATH + 'frame' + out_name + '.jpg')
        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(true_dist[:, 0], true_dist[:, 1], true_dist[:, 2], c='orange')
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='green')
            plt.savefig(TMP_PATH + 'frame' + out_name + '.jpg')
        
        # density plot side by side
        if plot_density:
            bins = 120
            fig, axs = plt.subplots(1, 2, figsize=(8,4), subplot_kw={'aspect': 'equal'})
            axs[0].hist2d(true_dist[:, 0], true_dist[:, 1],
                          range=[[-RANGE, RANGE], [-RANGE, RANGE]],
                          bins=bins, cmap=plt.cm.jet)
            axs[0].set_title('Target samples')
            axs[1].hist2d(samples[:, 0], samples[:, 1],
                          range=[[-RANGE, RANGE], [-RANGE, RANGE]],
                          bins=bins, cmap=plt.cm.jet)
            axs[1].set_title('Generator samples')
            plt.savefig(TMP_PATH + 'density' + out_name + '.jpg')
        
        frame_index[0] += 1


# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':

        dataset = []
        for i in range(100000 / 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) / BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':
        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            yield data

    elif DATASET == '8gaussians':
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
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .2
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = torch.Tensor(dataset)
            dataset /= 1.414  # stdev
            yield dataset
            
    elif DATASET == 'sine':
        while True:
            noise = 0.2
            x = torch.linspace(-4, 4, BATCH_SIZE, dtype=torch.float32)
            y = np.sin(x) + noise*np.random.randn(*x.shape)
            yield torch.stack([x, y], dim=1)
            
    elif DATASET == 'heteroscedastic':
        theta = torch.linspace(0, 2, BATCH_SIZE)
        x = np.exp(theta)*np.tan(0.1*theta)
        while True:
            b = (0.001 + 0.5 * np.abs(x)) * np.random.normal(1, 1, BATCH_SIZE)
            y = np.exp(theta)*np.sin(0.1*theta) + b
            yield torch.stack([x, y], dim=1)
            
    elif DATASET == 'moon':
        noise = 0.2
        while True:
            data, _ = sklearn.datasets.make_moons(n_samples=BATCH_SIZE,
                                                  noise=noise)
            yield torch.Tensor(data)
    
    elif DATASET == 'helix':
        noise = 0.2
        while True:
            t = torch.linspace(0, 20, BATCH_SIZE)
            x = np.cos(t)
            x2 = np.sin(t) + noise * np.random.randn(*x.shape)
    
            yield torch.stack([x, x2, t], dim=1)
    
    elif DATASET == 'circle':
        while True:
            t = np.random.random(BATCH_SIZE) * 2 * np.pi - np.pi
            length = 1 - np.random.random(BATCH_SIZE)*0.4
            x = torch.Tensor(np.multiply(np.cos(t), length))
            y = torch.Tensor(np.multiply(np.sin(t), length))
    
            yield torch.stack([x, y], dim=1)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    
    return gradient_penalty

def single_zero_grad(self):
    r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

def predict_fixed(netG, fixed_x, input_size, output_size):
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

    Returns
    -------
    None.

    """
    # placeholder
    true_dist = torch.zeros((input_size, INPUT_DIM), dtype=torch.float32)
    # fix x
    true_dist[:, 0] = torch.Tensor([fixed_x]*input_size)
    
    # mse loss
    mse = nn.MSELoss()
    noise = torch.randn(input_size, LATENT_DIM)
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
        samples = netG(noise, true_dist)
        loss = mse(samples[:, 0], true_dist[:, 0])
        if not min_loss or loss.item() < min_loss:
            min_loss = loss.item() 
            min_latent = noise.detach()
        losses.append(loss.item())
        loss.backward()
        optimizerL.step()
        
    # run second optimizer
    for epoch in range(30000):
        samples = netG(noise, true_dist)
        loss = mse(samples[:, 0], true_dist[:, 0])
        if not min_loss or loss.item() < min_loss:
            min_loss = loss.item() 
            min_latent = noise.detach()
        losses.append(loss.item())
        loss.backward()
        optimizerL1.step()
    
    # create samples using the optimimal latent space
    samples = netG(min_latent, true_dist).detach()
    xs = samples[:, 0]
    x = true_dist[0, 0].item()
    # sort by absolute value of the difference between 
    # generated x and fixed_x
    diff = abs(xs - x)  
    indices = torch.argsort(diff)
    indices_sub = indices[:output_size]
    
    return samples[indices_sub, :]

# ==================Definition End======================


# ==================Training Starts======================

# start writing log
f = open(TMP_PATH + "log.txt", "w")

netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)

# print model structures
f.write(str(netG))
f.write(str(netD))
f.write('\n')

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_train_gen()

losses = []
wass_dist = []

# start timing
start = timeit.default_timer()

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for iter_d in range(CRITIC_ITERS):
        _data = next(data).float()
        if use_cuda:
            _data = _data.cuda()

        netD.zero_grad()

        # train with real
        D_real = netD(_data)
        D_real = D_real.mean().unsqueeze(0)
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, LATENT_DIM)
        if use_cuda:
            noise = noise.cuda()
        fake = netG(noise, _data)
        D_fake = netD(fake.detach())
        D_fake = D_fake.mean().unsqueeze(0)
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, _data, fake)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    if not FIXED_GENERATOR:
        ############################
        # (2) Update G network
        ############################
        netG.zero_grad()

        _data = next(data).float()
        if use_cuda:
            _data = _data.cuda()

        noise = torch.randn(BATCH_SIZE, LATENT_DIM)
        if use_cuda:
            noise = noise.cuda()
        fake = netG(noise, _data)
        G = netD(fake)
        G = G.mean().unsqueeze(0)
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
    
    losses.append([G_cost.cpu().item(), D_cost.cpu().item()])
    wass_dist.append(Wasserstein_D.cpu().item())
    
    if iteration % 1000 == 999:
        # save discriminator model
        torch.save(netD.state_dict(), TMP_PATH + 'disc_model' + str(iteration) + '.pth')
        # save generator model
        torch.save(netG.state_dict(), TMP_PATH + 'gen_model' + str(iteration) + '.pth')
        # report iteration number
        f.write('Iteration ' + str(iteration) + '\n')
        # report time
        stop = timeit.default_timer()
        f.write('    Time spent: ' + str(stop - start) + '\n')
        # report loss
        f.write('    Generator loss: ' + str(G_cost.cpu().item()) + '\n')
        f.write('    Discriminator loss: ' + str(D_cost.cpu().item()) + '\n')
        f.write('    Wasserstein distance: ' + str(Wasserstein_D.cpu().item()) + '\n')
        # save frame plot
        generate_image(_data.cpu().numpy(), str(iteration),
                       plot_contour=plot_contour, plot_density=plot_density,
                       plot_3d=plot_3d)
        # save loss plot
        fig, ax = plt.subplots(1, 1, figsize=[20, 10])
        ax.plot(losses)
        ax.legend(['Generator', 'Discriminator'])
        plt.title('Generator Loss v.s Discriminator Loss')
        ax.grid()
        plt.savefig(TMP_PATH + 'loss_trend' + str(iteration) + '.png')
        # save wassertein loss plot
        fig, ax = plt.subplots(1, 1, figsize=[20, 10])
        ax.plot(wass_dist)
        plt.title('Wassertein Distance')
        ax.grid()
        plt.savefig(TMP_PATH + 'wass_dist' + str(iteration) + '.png')

# close log file
f.close()

# ==================Training Ends======================

# ==================Fixed Input Starts======================

# load pre-trained models and create fixed-input predictions.

# sine
netG.load_state_dict(torch.load(TMP_PATH + 'gen_model7999.pth'))
preds = None
for x in np.linspace(-4., 4., 17):
    print(x)
    out = predict_fixed(netG, x, 300, 14)
    if preds == None:
        preds = out
    else:
        preds = torch.cat((preds, out))
    plt.scatter(preds[:, 0], preds[:, 1])
    plt.show()
    
true_dist = next(data)
plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange')
plt.scatter(preds[:, 0], preds[:, 1], c='blue')
plt.savefig(TMP_PATH + 'fixed_input7999.jpg')


# heteroscedastic
netG.load_state_dict(torch.load(TMP_PATH + 'gen_model9999.pth'))
preds1 = None
for x in np.linspace(0, 1.4, 15):
    print(x)
    out1 = predict_fixed(netG, x, 300, 14)
    if preds1 == None:
        preds1 = out1
    else:
        preds1 = torch.cat((preds1, out1))
    plt.scatter(preds1[:, 0], preds1[:, 1])
    plt.show()

true_dist1 = next(data)
plt.scatter(true_dist1[:, 0], true_dist1[:, 1], c='orange')
plt.scatter(preds1[:, 0], preds1[:, 1], c='blue')
plt.savefig(TMP_PATH + 'fixed_input9999.jpg')

# sine
netG.load_state_dict(torch.load(TMP_PATH + 'gen_model9999.pth'))
preds2 = None
for x in np.linspace(-4., 4., 17):
    print(x)
    out2 = predict_fixed(netG, x, 300, 14)
    if preds2 == None:
        preds2 = out2
    else:
        preds2 = torch.cat((preds2, out2))
    plt.scatter(preds2[:, 0], preds2[:, 1])
    plt.show()
    
true_dist2 = next(data)
plt.scatter(true_dist2[:, 0], true_dist2[:, 1], c='orange')
plt.scatter(preds2[:, 0], preds2[:, 1], c='blue')
plt.savefig(TMP_PATH + 'fixed_input9999.jpg')

# sine
netG.load_state_dict(torch.load(TMP_PATH + 'gen_model2999.pth'))
preds3 = None
for x in np.linspace(-4., 4., 17):
    print(x)
    out3 = predict_fixed(netG, x, 300, 14)
    if preds3 == None:
        preds3 = out3
    else:
        preds3 = torch.cat((preds3, out3))
    plt.scatter(preds3[:, 0], preds3[:, 1])
    plt.show()
    
true_dist3 = next(data)
plt.scatter(true_dist3[:, 0], true_dist3[:, 1], c='orange')
plt.scatter(preds3[:, 0], preds3[:, 1], c='blue')
plt.savefig(TMP_PATH + 'fixed_input2999.jpg')


# circle 
netG.load_state_dict(torch.load(TMP_PATH + 'gen_model3999.pth'))
preds4 = None
for x in np.linspace(-0.8, 0.8, 17):
    print(x)
    out4 = predict_fixed(netG, x, 350, 20)
    if preds4 == None:
        preds4 = out4
    else:
        preds4 = torch.cat((preds4, out4))
    plt.scatter(preds4[:, 0], preds4[:, 1])
    plt.show()
    
true_dist4 = next(data)
plt.scatter(true_dist4[:, 0], true_dist4[:, 1], c='orange')
plt.scatter(preds4[:, 0], preds4[:, 1], c='blue')
plt.savefig(TMP_PATH + 'fixed_input3999.jpg')

# ==================Fixed Input Ends======================


# Miscellaneous
# plt.scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy())
# plt.scatter(_data[:, 0].detach().numpy(), _data[:, 1].detach().numpy())

