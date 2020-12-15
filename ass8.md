---
layout: default
title: Assignment 8
id: ass8
---


# Assignment 8: Autoencoders
**Deadline: January 4th, 11am**


In this assignment, you will implement various autoencoder architectures on our
beloved (Fashion) MNIST data. In particular, you will gain some insight into
the problem of training convolutional autoencoders.


## Autoencoders in Tensorflow

Building autencoders in Tensorflow is pretty simple. You need to define an
encoding based on the input, a decoding based on the encoding, and a loss
function that measures the distance between decoding and input. An obvious choice
may be simply the mean squared error (but see below). To start off, you could try
simple MLPs. Note that you are in no way obligated to choose the "reverse"
encoder architecture for your encoder; e.g. you could use a 10-layer MLP as an
encoder and a single layer as a decoder if you wanted. As a start, you can/should
opt for an "undercomplete" architecture where the encoding is smaller than the
data.

**Note:** The activation
function of the last decoder layer is very important, as it needs to be able to
map the input data range. Having data in the range [0, 1] allows you to use a
sigmoid output activation, for example. Experiment with different activations
such as sigmoid, relu or linear (no) activation and see how it affects the
model. Your loss function should also "fit" the output function, e.g. a sigmoid
output layer goes well with a binary (!) cross-entropy loss.


## Convolutional Autoencoders

Next, you should switch to a convolutional encoder/decoder to make use of the
fact that we are working with image data. The encoding should simply be one or
more convolutional layers, with any filter size and number of filters (you can
optionally apply fully-connected layers at the end). As an "inverse" of a
`Conv2D`, `Conv2DTranspose` is commonly used. However,
you could also use `UpSampling2D` along with regular convolutions.
 Again, there is no
requirement to make the parameters of encoder and decoder "fit", e.g. you don't
need to use the same filter sizes. However, you need to take care when choosing
 padding/strides such that the output has the same dimensions as the input. This
 can be a problem with MNIST (why?).
  It also means that the last
convolutional (transpose) layer should have as many filters as the input 
space (e.g. one filter for MNIST or three for CIFAR).


## What do Autoencoders Learn?

Keep in mind that the reconstruction loss is not a good proxy for the "quality" of an
autoencoder; instead you need to get an impression of what the model learned
about the input.
Note that if you use a single-layer (fully-connected) decoder, its weight
matrix will be `h_dim x 784` and each of the `h_dim` rows can be reshaped to
28x28 to get an impression of what kind of image the respective hidden
dimension represents.
The same holds for the encoder of course, which in the single-layer case will
have a `784 x h_dim` weight matrix. You should visualize some of your model's
filters to see what it learns.

Another way to interpret what a model has learned is by manipulating the code
space. This is particularly useful for deeper models where the above method is
of limited use. The idea is as follows: Take an image and encode it. Pick a
single encoding dimension (you could also pick multiple ones of course) and
change the value, then decode and inspect the resulting image. By attempting
many different values for the dimension (we may call this "walking the code
space"), we might be able to interpret what this particular variable encodes.
For example, gradually changing the value in one dimension might change the
stroke thickness in MNIST digits. Try this out for a few dimensions of a
trained model's code and see whether you can interpret their "meaning". This
will likely work better for models with small and/or sparse encodings.


## Unsupervised Pretraining

Autoencoders are useful in that they can learn from unlabeled data. This can
significantly improve performance in settings where large amounts of data are
available, but few labels. We can artificially evoke such a situation by just 
"pretending" that parts of our data has no labels. Try this:

- Train an autoencoder as before.
- Take a small random subset of the training data. Make sure it is actually random,
i.e. all labels are represented. You could go very low, e.g. 100 elements or so.
- Take the encoder part of the autoencoder. Stick a classifier on top and train
_only this layer_ on the subset of data. Your model will now learn a classifier
based on the _fixed_ features that were learned by the autoencoder.
- Compare performances (on the test set!) of the following approaches. Try to find
explanations for your observations.
  - Train autoencoder -- freeze encoder -- train classifier on top (as described above).
  - Train autoencoder -- train classifier on top of encoder. Do _not_ freeze the encoder,
    i.e. the encoder is "fine-tuned" on the labeled subset of data as well.
  - Train a classifier directly on the labeled subset; no pretraining. For fairness,
    it should have the same architecture as the encoder + classifier above.
    
Note: This will only "work" if your autoencoder has learned the "correct" level
of abstraction of the data. Else, its features might not actually be useful for
classification. The model is more likely to appropriate with proper regularization
on the latent space.


## Bonus: Going Beyond

There are many other things to try here. For example, you could try different
kinds of data, including non-image data. You have already worked with text,
although this is rarely used in an autoencoder context. Another option could be
audio data, e.g. the Tensorflow Speech Commands dataset. This is discussed
[here](https://www.tensorflow.org/tutorials/audio/simple_audio) in a classification
context, but the sections on preprocessing should be helpful.

Aside from that, you may try different kinds of autoencoders from the book:
- Sparse autoencoders
- Denoising
- Contractive
- Stochastic
- ...

Another interesting approach is offered by 
[winner-take-all autoencoders](https://arxiv.org/pdf/1409.2752.pdf), which use
an _extremely_ sparse encoding mechanism to learn very complex filters.


# Handing In
- Any and all experiments you try can be handed in, ideally with properly
documented results to get an overview of how different architectures perform etc.
- At the very least, provide a (convolutional) autoencoder on a dataset like MNIST
or CIFAR. Further, do at least some simple visualization/interpretation experiments
using techniques like visualizing learned weights or latent space walks. Finally,
try your hand at the unsupervised pretraining and report your results.