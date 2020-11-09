---
layout: default
title: Assignment 3
id: ass3
---


# Assignment 3: Keras & CNNs
**Deadline: November 16th, 11am**

In this assignment, you will get to know Keras, the high-level API in Tensorflow.
You will also create a better model for the MNIST dataset using
convolutional neural networks.


## Keras

The low-level TF functions we used so far are nice to have full control over
everything that is happening, but they are cumbersome to use when we just need
"everyday" neural network functionality. For such cases, Tensorflow has 
integrated Keras to provide abstractions over many
common workflows. Keras has _tons_ of stuff in it; we will only look at some of
it for this assignment and get to know more over the course of the semester.
In particular:

- `tf.keras.layers` provides wrappers for many common layers such as dense
(fully connected) or convolutional layers. This takes care of creating and
storing weights, applying activation functions, regularizers etc.
- `tf.keras.Model` in turn wraps layers into a cohesive whole that allows us to
handle whole networks at once.
- `tf.optimizers` make training procedures such as gradient descent simple.
- `tf.losses` and `tf.metrics` allow for easier tracking of model performance.

Unfortunately, none of the TF tutorials are _quite_ what we would like here, so
you'll have to mix-and-match a little bit:
- [This tutorial](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
covers most of what we need, i.e. defining a model and using it in a custom
training loop, along with optimizers, metrics etc. You can skip the part about
GANs. Overall, the loop works much the same as before, except:
  - You now have all model weights conveniently in one place.
  - You can use the built-in loss functions, which are somewhat less verbose
  than `tf.nn.sparse_softmax_cross_entropy_with_logits_and_many_other_things_also.`
  - You can use `Optimizer` instances instead of manually subtracting gradients
  from variables.
  - You can use `metrics` to keep track of model performance.
- There are several ways to build Keras models, the simplest one being 
`Sequential`. For additional examples, you can look at the top of 
[this tutorial](https://www.tensorflow.org/tutorials/keras/classification), or
[this one](https://www.tensorflow.org/tutorials/images/classification#create_the_model),
or maybe [this one](https://www.tensorflow.org/tutorials/images/cnn#create_the_convolutional_base)...
In each, look for the part `model = tf.keras.Sequential...`. You just put in a
list of layers that will be applied in sequence. Check 
[the API](https://www.tensorflow.org/api_docs/python/tf/keras/layers) to get an
impression of what layers there are and which arguments they take.
  
Later, we will see how to wrap entire model definitions, training loops and
evaluations in a hand-full of lines of code. For now, you might want to rewrite
your MLP code with these Keras functions and make sure it still works as before.

An example notebook can be found 
[here](https://ovgu-ailab.github.io/idl2020w/assignments/3/mlp_keras.ipynb).


## CNN for MNIST

You should have seen that (with Keras) modifying layer sizes, changing
activation functions
etc. is simple: You can generally change parts of the model without affecting
the rest of the program (training loop etc). In fact, you can change _the full 
pipeline from input to model output_ without having to change anything else 
(restrictions apply).

**Replace your MNIST MLP by a CNN.** The tutorials linked above might give
you some ideas for architectures. Generally:
- Your data needs to be in the format `width x height x channels`. So for MNIST,
make sure your images have shape `(28, 28, 1)`, not `(784,)`!
- Apply a bunch of `Conv2D` and possibly `MaxPool2D` layers.
- `Flatten`.
- Apply any number of `Dense` layers and the final classification (logits) layer.
- Use Keras!
- A reference CNN implementation _without_ Keras can be found 
[here](https://ovgu-ailab.github.io/idl2020w/assignments/3/conv_low_level.ipynb)!

 **Note:** Depending on your machine, 
training a CNN may take
*much* longer than the MLPs we've seen so far. Here, using Colab's GPU support
could be useful **(Edit -> Notebook settings -> Hardware Accelerator)**.
Also, processing the full test
set in one go for evaluation might be too much for your RAM. In that case, you
could break up the test set into smaller chunks and average the results 
(easy using keras metrics) -- or just make the model smaller.

You should consider using a better optimization algorithm than the basic
`SGD`. One option is to use adaptive algorithms, the most
popular of which is called Adam. Check out `tf.optimizers.Adam`. This will
usually lead to much faster learning without manual tuning of the learning rate
or other parameters. We will discuss advanced optimization strategies later in
the class, but the basic idea behind Adam is that it automatically
chooses/adapts a per-parameter learning rate as well as incorporating momentum.
Using Adam, your CNN should beat your MLP after only a few hundred steps of
training. The
general consensus is that a well-tuned gradient descent with momentum and
learning rate decay will outperform adaptive methods, but you will need to
invest some time into finding a good parameter setting -- we will look into
these topics later.

If your CNN is set up well, you should reach extremely high accuracy results.
This is arguably where MNIST stops being interesting. If you haven't done so,
consider working with Fashion-MNIST instead (see
[Assignment 1](https://ovgu-ailab.github.io/idl2020w/ass1.html)). This should
present more of a challenge and make improvements due to hyperparameter tuning
more obvious/meaningful. You could even try CIFAR10 or CIFAR100 as in one of the
tutorials linked above. They have 32x32 3-channel color images with much more
variation. These datasets are also available in `tf.keras.datasets`.  
**Note:** For some reason, the CIFAR labels are organized somewhat differently --
shaped `(n, 1)` instead of just `(n,)`. You should do something like
`labels = labels.reshape((-1,))` or this will mess up the loss function.


## What to Hand In

- A CNN (built with Keras) trained on MNIST (or not, see below). Also use Keras
losses, optimizers and metrics, but do still use a "custom" training loop
(with `GradientTape`).
- You are _highly_ encouraged to move past MNIST at this point. E.g. switching
to CIFAR takes minimal effort since it can also be downloaded through Keras. You
can still use MNIST as a "sanity check" that your model is working, but you can
skip it for the submission.
- Document any experiments you try. For example:
  - Really do play with the model parameters. As a silly example, you could try
increasing your filter sizes up to the input image size -- think about what kind
of network you are ending up with if you do this! On the other extreme, what about
1x1 filters?
    - You can do the same thing for pooling. Or replace pooling with strided
  convolutions. Or...
  - If you're bored, just try to achieve as high of a (test set) performance as you
can on CIFAR. This dataset is still commonly used as a benchmark today. Can you
get ~97%?
  - You could try to "look into" your trained models. E.g. the convolutional layers
output "feature maps" that can also be interpreted as images (and thus plotted
-- one image per filter).
You could use this to try to figure out what features/patterns the filters are
recognizing by seeing for what inputs they are most active.
