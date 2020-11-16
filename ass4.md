---
layout: default
title: Assignment 4
id: ass4
---


# Assignment 4: Graphs & DenseNets
**Deadline: November 23th, 11am**


**Note: Find the notebook from the exercises 
[here](https://ovgu-ailab.github.io/idl2020w/assignments/4/function_al.ipynb).**

## Graph-based Execution

So far, we have been using so-called "eager execution" exclusively: Commands are
run as they are defined, i.e. writing `y = tf.matmul(X, w)` actually executes
the matrix multiplication.

In Tensorflow 1.x, things used to be different: Lines like the above would only
_define the computation graph_ but not do any actual computation. This would be
done later in dedicated "sessions" that execute the graph. Later, eager 
execution was added as an alternative way of writing programs and is now the
default, mainly because it is much more intuitive/allows for a more natural
workflow when designing/testing models.

Graph execution has one big advantage: It is very efficient because entire
models (or even training loops) can be executed in low-level C/CUDA code without
ever going "back up" to Python (which is slow). As such, TF 2.0 still retains
the possibility to run stuff in graph mode if you so wish -- let's have a look!

It seems like the only tutorial on the TF website right now is 
[this one](https://www.tensorflow.org/tutorials/customization/performance)
which goes intro extreme depth on all the subtleties. The basic gist is:
- You can annotate a Python function with `@tf.function` to "activate" graph
execution for this function.
- The first time this function is called, it will be _traced_ and converted to
a graph.
- Any other time this function is called, _the Python function will not be run;
instead the traced graph is executed_.
- The above is not entirely true -- functions may be _retraced_ under certain
(important) conditions, e.g. for every new "input signature". This is treated in
detail in the article linked above.
- Beware of using Python statements like `print`, these will not be traced so
the statement will only be called during the tracing run itself. If you want to
print things like tensor values, use `tf.print` instead. Basically, traced TF
functions only do "tensor stuff", not general "Python stuff".

Go back to some of your pevious models and sprinkle some `tf.function` annotations
in there. You might need to refactor slightly -- you need to actually wrap things
into a function!
- The most straightforward target for decoration is a "training step" function
that takes a batch of inputs and labels, runs the model, computes the loss and
the gradients and applies them.
- In theory, you could wrap a whole training loop (including iteration over a
dataset) with a `tf.function`. If you can get this to work on one of your
previous models _and actually get a speedup_, you get a cookie. :)


## DenseNet

Previously, we saw how to build neural networks in a purely sequential manner --
each layer receives one input and produces one output that serves as input to
the next layer. There are many architectures that do not follow this simple
scheme. You might ask yourself how this can be done in Keras. One answer is via
the so-called functional API. There is an in-depth guide 
[here](https://www.tensorflow.org/guide/keras/functional). Reading just the intro
should be enough for a basic grasp on how to use it, but of course you can read
more if you wish.

Next, use the functional API to implement a [DenseNet](https://arxiv.org/pdf/1608.06993.pdf).
You do _not_ need to follow the exact same architecture, in fact you will probably
want to make it smaller for efficiency reasons. Just make sure you have one or
more "dense blocks" with multiple layers (say, three or more) each. 
You can also leave out batch
normalization (this will be treated later in the class) as well as "bottleneck
layers" (1x1 convolutions) if you want.

Bonus: Can you implement DenseNet with the Sequential API? You might want to look
at how to [implement custom layers](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
(shorter version [here](https://www.tensorflow.org/tutorials/customization/custom_layers))...


## What to Hand In

- DenseNet. Thoroughly experiment with (hyper)parameters. Try to achieve the best
performance you can on CIFAR10/100.
- If the above is really too much for you, you _can_ settle for re-implementing
old models with the functional API and `tf.function`. For the latter, report
performance (in terms of speed) when using the decorator vs. leaving it out.
However, the point of this assignment really is the DenseNet -- in that case,
you can leave these more basic things out of your submission.
- Reminder for the cookie mentioned above. ;)

The next two parts are just here for completeness/reference, to show other ways 
of working with Keras and some additional TensorBoard functionalities. Check
them out if you want.


## Bonus: High-level Training Loops with Keras

As mentioned previously, Keras actually has ways of packing entire training loops
into very few lines of code. This is good whenever you have a fairly "standard"
task that doesn't require much besides iterating over a dataset and computing a
loss/gradients at each step. In this case, you don't need the customizability
that writing your own training loops gives you.

As usual, here are some tutorials that cover this:
- The gist is covered in [the beginner quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner):
Build the model, compile with an optimizer, a loss and optional metrics and then
run `fit` on a dataset. That's it!
- They also have [the same thing with a bit more detail](https://www.tensorflow.org/tutorials/keras/classification).
- The above covers the bare essentials, but you could also look at 
[how to build a CNN for CIFAR10](https://www.tensorflow.org/tutorials/images/cnn).

There are also some interesting overview articles in the "guide" section but this
should suffice for now. Once again, go back to your previous models and re-make
them with these high-level training loops! Also, from now on, feel free to run
your models like this if you want (and can get it to work for your specific case).


## Bonus: TensorBoard Computation Graphs

You can display the computation graphs Tensorflow uses internally in TensorBoard.
This can be useful for debugging purposes as well as to get an impression what
is going on "under the hood" in your models. More importantly, this can be combined
with _profiling_ that lets you see how much time/memory specific parts of your
model take.

To look at computation graphs, you need to _trace_ computations explicitly.
See the last part of [this guide](https://www.tensorflow.org/tensorboard/graphs#graphs_of_tffunctions)
for how to trace `tf.function`-annotated computations. Note: It seems like you
have to do the trace the first time the function is called (e.g. on the first
training step).
