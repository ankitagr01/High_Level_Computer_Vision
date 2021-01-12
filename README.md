# High_Level_Computer_Vision

## Assignment 1: Image Filtering and Object Identification

In this exercise you will first familiarise yourself with basic image filtering routines. In the second part, you will develop a simple image querying system which accepts a query image as input and then finds a set of similar images in the database. In order to compare images you will implement some simple histogram based distance functions and evaluate their performance in combination with different image representations.

## Assignment 2: Deep Neural Networks and Backpropagation

Deep neural networks have shown staggering performances in various learning tasks, including computer vision, natural language processing, and sound processing. They have made the model designing more  exible by enabling end-to-end training.
In this exercise, we get to have a first hands-on experience with neural network training. Many frameworks (e.g. PyTorch, Tensorflow, Caffe) allow easy usage of deep neural networks without precise knowledge on the inner workings of backpropagation and gradient descent algorithms. While these are very useful tools, it is important to get a good understanding of how to implement basic network training from scratch, before using this libraries to speed up the process. For this purpose we will implement a simple two-layer neural network and its training algorithm based on back-propagation using only basic matrix operations in questions 1 to 3. In question 4, we will use a popular deep learning library, PyTorch, to do the same and understand the advantanges offered in using such tools.
As a benchmark to test our models, we consider an image classification task using the widely used CIFAR-10 dataset. This dataset consists of 50000 training images of 32x32 resolution with 10 object classes, namely airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The task is to code and train a parametrised model for classifying those images. This involves

- Implementing the feedforward model.
- Implementing the backpropagation algorithm (gradient computation).
- Training the model using stochastic gradient descent and improving the model training with better hyperparameters.
- Using the PyTorch Library to implement the above and experiment with deeper networks.

## Assignment 3: Convolutional Networks

In exercise 3, you will implement a convolutional neural network to perform image classification and explore methods to improve the training performance and generalization of these networks. 
We will use the CIFAR-10 dataset as a benchmark for out networks similar to the previous exercise. This dataset consists of 50000 training images of 32x32 resolution with 10 object classes, namely airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The task is to implement convolutional network to classify these images using the PyTorch library. The four questions are
- Implementing the convnet, train it and visualizing its weights.
- Experiment with batch normalization and early stopping.
- Data augmentation and dropout to improve generalization.
- Implement transfer learning from a ImageNet pretrained model.
