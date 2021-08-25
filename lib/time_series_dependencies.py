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
from sklearn import preprocessing
import pandas as pd

#################################
# Time series specific functions
#################################

def plot_time_data(fake_data, out_name, out_path, scaler):
    """
    Plots the generated data distribution and true data distribution. 
    Save plots. 

    Parameters
    ----------
    fake_data : torch.Tensor
        Generated samples.
    out_name : string
        Output suffix.
    out_path : string
        Output path.
    scaler : object
        Scaler standardization used to process data.

    Returns
    -------
    None.

    """
    # transform the scaled data back to real scale 
    # true_data = scaler.inverse_transform(true_data)

        # noise = torch.randn(150, LATENT_DIM)
        # fake_data = netG(noise).detach().numpy().reshape(150, ntimes, nvars)
        # fake_data = scaler.inverse_transform(fake_data.reshape(150*ntimes, nvars))
    
    # plot pm10
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c='blue', label='WGAN')
    # plt.scatter(true_data[:, 0], true_data[:, 1], c='orange', label='Real data')
    plt.title('PM10')
    ax.legend()
    plt.savefig(out_path + 'pm10' + out_name + '.jpg')
    
     # plot pm2.5
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(fake_data[:, 0], fake_data[:, 2], c='blue', label='WGAN')
    # plt.scatter(true_data[:, 0], true_data[:, 2], c='orange', label='Real data')
    plt.title('PM2.5')
    ax.legend()
    plt.savefig(out_path + 'pm25' + out_name + '.jpg')

     # plot pm1
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(fake_data[:, 0], fake_data[:, 3], c='blue', label='WGAN')
    # plt.scatter(true_data[:, 0], true_data[:, 3], c='orange', label='Real data')
    plt.title('PM1')
    ax.legend()
    plt.savefig(out_path + 'pm1' + out_name + '.jpg')
    
     # plot co2
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(fake_data[:, 0], fake_data[:, 4], c='blue', label='WGAN')
    # plt.scatter(true_data[:, 0], true_data[:, 4], c='orange', label='Real data')
    plt.title('CO2')
    ax.legend()
    plt.savefig(out_path + 'co2' + out_name + '.jpg')
    
     # plot temperature
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(fake_data[:, 0], fake_data[:, 5], c='blue', label='WGAN')
    # plt.scatter(true_data[:, 0], true_data[:, 5], c='orange', label='Real data')
    plt.title('Temperature (F)')
    ax.legend()
    plt.savefig(out_path + 'temp' + out_name + '.jpg')

     # plot rh
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(fake_data[:, 0], fake_data[:, 6], c='blue', label='WGAN')
    # plt.scatter(true_data[:, 0], true_data[:, 6], c='orange', label='Real data')
    plt.title('Relative Humidity')
    ax.legend()
    plt.savefig(out_path + 'rh' + out_name + '.jpg')
        

def plot_time_predictions(true_data, fake_data, out_name, out_path, scaler):
    """
    Plots the fixed-input predictions against real data distribution. 

    Parameters
    ----------
    true_data : torch.Tensor
        True data.
    fake_data : torch.Tensor
        Generated samples.
    out_name : string
        Output suffix.
    out_path : string
        Output path.
    scaler : object
        Scaler standardization used to process data.    

    Returns
    -------
    None.

    """
    true_data = scaler.inverse_transform(true_data)
    fake_data = scaler.inverse_transform(fake_data)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(true_data[:, 0], fake_data[:, 1], c='blue', label='WGAN')
    plt.scatter(true_data[:, 0], true_data[:, 1], c='orange', label='Real data')
    plt.title('PM10')
    ax.legend()
    plt.savefig(out_path + 'pm10' + out_name + '_pred' + str(end_index) + '.jpg')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(true_data[:, 0], fake_data[:, 2], c='blue', label='WGAN')
    plt.scatter(true_data[:, 0], true_data[:, 2], c='orange', label='Real data')
    plt.title('PM2.5')
    ax.legend()
    plt.savefig(out_path + 'pm25' + out_name + '_pred' + str(end_index) + '.jpg')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(true_data[:, 0], fake_data[:, 3], c='blue', label='WGAN')
    plt.scatter(true_data[:, 0], true_data[:, 3], c='orange', label='Real data')
    plt.title('PM1')
    ax.legend()
    plt.savefig(out_path + 'pm1' + out_name + '_pred' + str(end_index) + '.jpg')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(true_data[:, 0], fake_data[:, 4], c='blue', label='WGAN')
    plt.scatter(true_data[:, 0], true_data[:, 4], c='orange', label='Real data')
    plt.title('CO2')
    ax.legend()
    plt.savefig(out_path + 'co2' + out_name + '_pred' + str(end_index) + '.jpg')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(true_data[:, 0], fake_data[:, 5], c='blue', label='WGAN')
    plt.scatter(true_data[:, 0], true_data[:, 5], c='orange', label='Real data')
    plt.title('T')
    ax.legend()
    plt.savefig(out_path + 'temp' + out_name + '_pred' + str(end_index) + '.jpg')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.scatter(true_data[:, 0], fake_data[:, 6], c='blue', label='WGAN')
    plt.scatter(true_data[:, 0], true_data[:, 6], c='orange', label='Real data')
    plt.title('RH')
    ax.legend()
    plt.savefig(out_path + 'rh' + out_name + '_pred' + str(end_index) + '.jpg')


def read_data(data_path):
    """
    Reads from csv the pollution data of a single day.

    Parameters
    ----------
    data_path : string
        Path and name of input data file

    Returns
    -------
    None.

    """
    print("Reading data file: " + data_path)
    raw = pd.read_csv(data_path)
    # clean data
    # keep only non-na values 
    raw.date = pd.to_datetime(raw.date)
    # raw = raw.fillna(method='ffill').fillna(method='bfill')
    raw.columns = ['date', 'PM10', 'PM25', 'PM1', 'CO2', 'T', 'RH']
    raw['minute_of_day'] = raw.date.dt.hour*60 + raw.date.dt.minute
    df = raw.dropna()
    df = df[['minute_of_day', 'PM10', 'PM25', 'PM1', 'CO2', 'T', 'RH']]
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    np_out = min_max_scaler.fit_transform(df.values)
    return np_out, min_max_scaler


def concat_timesteps(X_time, ntimes, step, times):
    """
    Create time windows from time series data

    Parameters
    ----------
    X_time : numpy array
        Time series data.
    ntimes : int
        Length of window.
    step : int
        Number of steps to skip.
    times : int
        Number of windows.

    Returns
    -------
    numpy array
        An array of consecutive time windows constructed from the input data.

    """
    X_time_concat = []
    for j in range(len(X_time)//times):
        for i in range(j*times, j*times+(times-ntimes*step)):
            X_time_concat.append(X_time[i:i+ntimes*step:step])
    return np.array(X_time_concat)


def predict_fixed(netG, true_dist, input_size, output_size):
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
        A pre-trained generator object.
    true_dist : numpy array
        True data.
    input_size : int
        How many values to generate. Must be greater than output_size.
    output_size : int
        How many predictions to make.

    Returns
    -------
    numpy array
        The predictions generated by GAN

    """
    # fix x
    true_dist = torch.Tensor(np.repeat(true_dist.reshape((1, nvars*ntimes)), input_size, 0))
    time_index = [int(i) for i in list(np.linspace(0, 63, 10))]
    
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
    
    # run optimizer
    for epoch in range(10000):
        samples = netG(noise)
        loss = mse(samples[:, :(INPUT_DIM-nvars+2)], true_dist[:, :(INPUT_DIM-nvars+2)])
        if not min_loss or loss.item() < min_loss:
            min_loss = loss.item() 
            min_latent = noise.detach()
        losses.append(loss.item())
        loss.backward()
        optimizerL.step()
            
    # create samples using the optimimal latent space
    samples = netG(min_latent).detach()
    # # use true time values
    samples[:, time_index] = true_dist[:, time_index]
    xs = samples[:, :(INPUT_DIM-nvars)]
    x = true_dist[:, :(INPUT_DIM-nvars)]
    # sort by absolute value of the difference between 
    # generated x and fixed_x
    diff = ((xs - x)**2).mean(1)
    indices = torch.argsort(diff)
    indices_sub = indices[:output_size]
    
    return samples[indices_sub, (INPUT_DIM-nvars+2):]


def predict_fixed_continuous(df, start_time, nsteps):
    """
    Continuously makes predictions for time series data. Uses previous predictions
    to predict for later time levels. 

    Parameters
    ----------
    df : numpy array
        True data. The first few time levels will be used to start the process. 
        Time and PM 10 will be specified (no predictions). 
    start_time : int
        The beginning time level.
    nsteps : int
        How many predictions to make.

    Returns
    -------
    numpy array
        The predicted time series data.

    """
    # output_window > ntimes
    pred_out = np.zeros((ntimes + nsteps - 1, nvars))
    pred_out[:ntimes, :] = df[start_time:(start_time + ntimes), :]
    # specify time and pm10 
    pred_out[:, :2] = df[start_time:(start_time + ntimes + nsteps - 1), :2]
    for step in range(nsteps):
        # print("step " + str(step))
        # print("updating " + str(ntimes+step - 1))
        # print("using data from " + str(step) + " to " + str(ntimes + step - 1))
        pred = predict_fixed(netG, pred_out[step:(ntimes + step), :], 10, 1)
        pred_out[(ntimes + step - 1), 2:] = pred.detach().numpy()
    return np.maximum(pred_out, 0)



