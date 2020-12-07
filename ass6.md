---
layout: default
title: Assignment 6
id: ass6
---


# Assignment 6: More Realistic Language Modeling & Recurrent Neural Networks
**Deadline: December 7th, 11am**

In this task, we will once again deal with the issue of language modeling. This
time, however we will be using Tensorflow's RNN functionalities, which makes
defining models significantly easier. On the flipside, we will be dealing with
issues that come with variable-length inputs. This in turn makes defining
models significantly more complicated.
We will stick with character-level models for now; while word models are more
common in practice, they come with additional problems.


## Preparing the Data

Once again we provide you with a script to help you process raw text into a
format Tensorflow can understand. Download the script
[here](https://ovgu-ailab.github.io/idl2020w/assignments/6/prepare_data2.py).
This script differs from the previous one in a few regards:

- No more fixed sequence lengths. Instead, you need to provide a regular
expression which will be used as a sequence delimiter. You could just use the
newline character `\n` as a simple baseline (you might need to provide `\\n`
instead depending on your OS -- check whether the resulting sequence
lengths/number of sequences is reasonable. It looks like `\n` works with Windows
and `\\n` with Linux/Mac -- be aware that Colab runs a Linux environment). 
More interesting results should
come from a sensible delimiter for the given corpus. For example, try
`[0-9]+:[0-9]+` for the King James Bible (available 
[here](https://github.com/ErikSchierboom/sentencegenerator/tree/master/samples)) 
to split the text
into verses. Or try `\n\n+` (or `\\n\\n+`, see above) on the Shakespeare text
to split it into monologues. Both should give you about 30,000 sequences each,
with lengths peaking around 100 characters.
- Every sequence now ends with a special end-of-sequence character that allows
the model to learn how long a typical sequence should be.
- You can (and should) supply a `maxlen` argument that will remove any
sequences longer than this threshold. Useful to remove things such as the
Project Gutenberg disclaimers and generally keep your program from exploding
due to overly long inputs. For the above examples, a maximum length of 500 or
so seems reasonable.

The issue of differing sequence lengths also means that you need to provide 
the data to Tensorflow in a slightly different way during training:
At the end of the day, *Tensorflow* works on *tensors*, and these have fixed
sizes in each dimension. That is, sequences of different lengths can't be put
in the same tensor (and thus not in the same batch). The standard work-around
for this issue is *padding*: Sequences are filled up with "dummy values" to get
them all to the same length (namely, that of the longest sequence of the
batch). The most straightforward approach is to simply add these dummy values
at the end, and the most common value to use for this is 0. Doing padding is
simple in Tensorflow: Use `padded_batch` instead of `batch` in `tf.data`.


## Building an RNN

Defining an RNN is much simpler when using the full Tensorflow/Keras library.
There are tutorials available [here](https://www.tensorflow.org/tutorials/text/text_generation)
and [here](https://www.tensorflow.org/guide/keras/rnn),
 but these are the basic steps:

### Way 1: Fully pre-built
Classes like `tf.keras.layers.SimpleRNN/LSTM/GRU` define a "full" RNN: These
classes expect 3D inputs (i.e. batch, time, feature axes) and run the whole
sequence through the network in one go.  This can be extremely efficient because
the whole thing can be implemented as one "operation" in Tensorflow. However,
you are somewhat limited with regard to customizability.

### Way 2: Using Cells
Another way to define an RNN is by first defining a "cell", e.g. 
`tf.keras.layers.SimpleRNNCell` or `LSTMCell`, and then define a 
`tf.keras.layers.RNN` where you put in this cell. Basically, the cell defines
the computations _at one time step_ and the RNN wraps this into a computation
over a sequence. This allows for more control over what the cell does, but tends
to be less efficient.  
Note that you can build a "deep RNN" by putting a list of cells into `RNN`.

### Putting it together
No matter how you built your RNN, you should integrate it in a keras `Model`
along with a dense output layer mapping to the vocabulary again (this is _not_ part
of the Keras RNNs). Also, you will want `return_sequences` to be `True` in your 
RNN since we
want to get output probabilities for the next character at _each_ time step.

- You can also build deep RNNs by having multiple ones in a Keras sequential model.
- You may be confused about what a `Dense` layer does when applied to a 3D input. In
fact, it will simply be applied to the last dimension only, i.e. independently
to each time step. This is exactly what we want.
- Similarly, you can simply apply Tensorflow/Keras cross-entropy loss functions
to the outputs/labels as they are (including the time axis) and it will work
just fine (this will average over both time and batch axes).

**The very least you should do is to re-implement the task from last assignment
with these functionalities.** That is, you may work with fixed, known sequence
lengths as a start. However, the real task lies ahead and you may skip the
re-implementation and go straight for that one if you wish.


## Dealing with Variable-length Sequences

You may have noticed that there is one problem remaining: We padded shorter
sequences to get valid tensors, but the RNN functions as well as the cost
computations have no way of actually knowing that we did this. This means that
the network will "learn" on the padding elements as well, which will artificially
lower the loss and divert the training process from optimizing on those parts
of the data that really count (the not-padding). Let's **fix these issues.**

- In principle, you can use Keras functionality for this, as described 
[here](https://www.tensorflow.org/guide/keras/masking_and_padding#mask-generating_layers_embedding_and_masking).
However, we will keep things a bit more low-level for now to make sure you
understand the concepts at work here.
- The first "issue" is that the outputs for padded inputs are going to be
nonsense. However, this is not _really_ an issue since padding appears only at
the end of sequences, and thus we can simply ignore the corresponding outputs
when computing the loss. Question: There may be scenarios where it's important
that padded inputs are ignored, i.e. no computation is done/passed through on
them. Can you think of a way to implement this? You may assume low-level control
over what happens at each time step, like in the last asignment.
- To do this "ignoring", we compute a _mask_. This is simply a tensor of 1s and
0s, with 0 coding an element that we want to "mask out".
  - Since we use 0 as padding value, `tf.math.count_nonzero` can be used to count,
  for each sequence, the number of "real" (non-padding) elements. This should
  just be a 1D tensor, one count per sequence in the batch. Note that you
  should _subtract 1 from these_ since the last element of each sequence isn't
  used as input.
  - Then, `tf.sequence_mask` can turn this into a mask (2D; `batch x time`). You
  will want this to be `float32`; the default is `bool`.
  - Alternatively, you could use `tf.not_equal` to directly compare the values
  to 0.
- Having the mask available, you can mask out "invalid" values in the loss by
simple element-wise multiplication. You should use the softmax loss functions 
in the `tf.nn` module
because the Keras losses automatically summarize over the batch and you need
to do the masking _before_ this happens. That is, you need a `batch x time` loss
tensor to multiply with your `batch x time` mask.
- Finally, we need to aggregate the per-element costs into a single scalar cost. As in
the last assignment, we could either use a sum or average. This time, it
actually makes a difference due to the variable-length sequences! If averaging,
you cannot just use `tf.reduce_mean`: Note that the mean is just the sum
divided by the number of elements, and this number would also include padding
elements! Instead, you should use `reduce_sum` and then divide by the number of
"real" elements (remember the mask!), which differs for each element of the
batch.

While this was a lot of explanation, your program should hopefully be more
succinct than the previous one, and it's more flexible as well!
Experiment with different cell types or even multiple layers and see the
effects on the cost. Be prepared for significantly longer training times than
with feedforward networks like CNNs.


## Sampling with Keras

Sampling with Keras works a lot like before:
We can input single characters by simply treating
them as length-1 sequences. The process should be stopped not after a certain
amount of steps, but when the special character `</S>` is sampled (you could
also continue sampling and see if your network breaks down...). However, we have
a bit of a problem now because the Keras RNNs take a whole sequence as input at
once, but we want to _generate_ the sequence ourselves. Here's what you can do:

- Make your RNN _stateful_ (check the API for arguments) -- this will essentially
"preserve" the state of the network after an input sequence has been processed,
so that when the next input is received, it will use this stored state instead
of the initial state.
- With a stateful RNN, you can do pretty much the same thing as before, except
you do not have to supply the updated state to the network because this is saved
internally. Just generate and feed in characters one by one!
- During training, make sure to reset the states (use `Model.reset_states()`) after
each batch because you probably don't want the states to propagate between batches.
You should also reset states before starting to generate a sequence so that you
"start fresh" each time!
- Stateful RNNs _always_ need the same batch size, so make sure there are no
smaller "leftover" batches during training. Also, you will need to use the same
batch size for generation. Alternatively, you could create a new RNN for generation
that takes the same weights as the old one, but uses a different batch size.


## Applying a Language Model

Finally, keep in mind that language modeling is actually about assessing the
"quality" of a piece of language, usually formalized via probabilities. The
probability of a given sequence is simply the product of the probability
of each character (i.e. the softmax output of the RNN, for the character that
actually appeared) given the previous characters. Try this
out: **Compute the probabilities for some sequences typical for the training
corpus (you could just take them straight from there). Compare these to the
probabilities for some not-so-typical sequences.** Note that only sequences of
the same length can be compared since longer sequences automatically receive a
lower probability. For example, in the King James corpus you could simply
replace the word "LORD" by "LOOD" somewhere and see how this affects the
probability of the overall sequence.  
Note that for numerical reasons, you will generally want to work with _log
probabilities_ instead of "regular" ones.
