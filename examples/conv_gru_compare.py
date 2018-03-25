""" This script demonstrates the use of a convolutional GRU network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import os
import h5py
from keras.models import Sequential, load_model
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvGRU2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pylab as plt

# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    np.random.seed(0)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

# Create a data set
noisy_movies, shifted_movies = generate_movies(n_samples=1200)

# Load a trained models
modelPath1 = os.getcwd() + '/conv_gru_model_skip_connection.h5'
model1 = load_model(modelPath1)

modelPath2 = os.getcwd() + '/conv_gru_model_sum_bidirectional_merge.h5'
model2 = load_model(modelPath2)

modelPath3 = os.getcwd() + '/conv_gru_model_one_bid.h5'
model3 = load_model(modelPath3)

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1017
track = noisy_movies[which][:7, ::, ::, ::]
track2 = noisy_movies[which][:7, ::, ::, ::]
track3 = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = model1.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

    new_pos2 = model2.predict(track2[np.newaxis, ::, ::, ::, ::])
    new2 = new_pos2[::, -1, ::, ::, ::]
    track2= np.concatenate((track2, new2), axis=0)

    new_pos3 = model3.predict(track3[np.newaxis, ::, ::, ::, ::])
    new3 = new_pos3[::, -1, ::, ::, ::]
    track3 = np.concatenate((track3, new3), axis=0)

# And then compare the predictions
# to the ground truth
track1 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(222)

    if i >= 7:
        ax.text(1, 3, 'skip-connection', fontsize=10, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=10)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)

    ax = fig.add_subplot(223)

    if i >= 7:
        ax.text(1, 3, 'bidirectional', fontsize=10, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=10)

    toplot = track2[i, ::, ::, 0]

    plt.imshow(toplot)

    ax = fig.add_subplot(224)

    if i >= 7:
        ax.text(1, 3, 'one-bidirectional', fontsize=10, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=10)

    toplot = track3[i, ::, ::, 0]

    plt.imshow(toplot)

    ax = fig.add_subplot(221)
    plt.text(1, 3, 'Ground truth', fontsize=10)

    toplot = track1[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%s/%i_animate.png' % (os.getcwd(), (i + 1)))
