# Sketch Classification Model
by Joseph Savage

## Problem Overview
I decided to create a simple classifier with a UI that allows a sketch to be drawn, and an output prediction to be given.

As this is a toy problem that will eventually be explored further, I started with only 10 classes from the [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset.

The supported classes include: cat, apple, key, bed, basketball, cake, cloud, crown, duck, fish.

## Model Architecture
A simple CNN structure was designed, with 3 convolutional layers (with ReLU and max pooling at each layer) followed by 2 linear layers for classification.
