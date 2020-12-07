---
layout: default
title: Assignment 7
id: ass7
---

# Assignment 7: Attention-based Neural Machine Translation
**Deadline: December 14th, 11am**

In this task, you will implement a simple NMT with attention for a language pair 
of your choice.  
We will follow the corresponding 
[TF Tutorial on NMT](https://www.tensorflow.org/tutorials/text/nmt_with_attention).  

Please do **not** use the exemplary English-Spanish example to reduce temptation
 of simply copying the tutorial.  
You can find data sets [here](http://www.manythings.org/anki/). We recommend
 picking a language pair where you understand both languages (so if you do speak 
 Spanish... feel free to use it ;)). 
This makes it easier (and more fun) for you to evaluate the results.

You may run into issues with the code in two places:
1. The downloading of the data inside the notebook might not work (it crashes
with a 403 Forbidden error). In that case, you can simply download & extract the
data on your local machine and upload the .txt file to your drive, and then mount
it and load the file as you've done before.
2. The `create_dataset` function might crash. It expects each line to result in
*pairs of sentences*, but there seems to be a third element which talks about
attribution of the example. If this happens, you can use `l.split('\t')[:-1]` to
exclude this in the function.

Recommendation: Start with a small number of training examples. Use one of the 
training examples to evaluate whether training worked properly. 
Only switch to the complete data set if you're sure that your code works, 
because training is quite slow.  
Note that many words are out-of-vocabulary (OOV) when using few examples, and the
code will crash if attempting to translate sentences with unknown words. As a 
bonus, you could try adapting the code to handle unknown words (e.g. introducing
a catch-all "UNKNOWN" token).

Tasks:
- Follow the tutorial and train the model on your chosen language pair 
(using Bahdanau attention).
- You might need to adapt the preprocessing depending on the language.
- Implement other attention mechanisms and train models with them>
  - dot product attention <img src="https://latex.codecogs.com/svg.latex?h_t^T%20\cdot%20\overline{h}_s" />
  - Luong's multiplicative attention <img src="https://latex.codecogs.com/svg.latex?h_t^T%20W%20\overline{h}_s" />

Hint: Storing the models is important, so you don't need to retrain them all the time.  
Also take care to not overwrite model checkpoints when switching from additive to multiplicative attention.

Compare the attention weight plots for some examples between the attention 
mechanisms.  
We recommend to add `,clim=[0,1]` when creating the plot in 
`ax.matshow(attention, cmap='viridis')` so the colors correspond to the same 
attention values in different plots. 
Note that the tutorial crops off the padding in the attention plot although the 
decoder can attend to those too.  
- Do you see qualitative differences in the attention weights between different
 attention mechanisms?  
- Do you think that the model attends to the correct tokens in the input language
 (if you understand both languages)?

Here are a few questions for you to check how well you understood the tutorial.  
Please answer them (briefly) in your solution!  
- Which parts of the sentence are used as a token? Each character, each word, 
or are some words split up?
- Do the same tokens in different language have the same ID?   
  e.g. Would the same token index map to the German word `die` and to the 
  English word `die`?
- What is the relation between the encoder output and the encoder hidden state
 which is used to initialize the decoder hidden state 
 (for the architecture used in the tutorial)?
- Is the decoder attending to all previous positions, including the previous
 decoder predictions?
- Does the Encoder output change in different decoding steps?
- Does the context vector change in different decoding steps?
- The decoder uses teacher forcing. Does this mean the time steps can be computed
 in parallel?
- Why is a mask applied to the loss function?

Bonus1: Can you prevent the attention mechanism from attending to padded
 positions in the sequence?

Bonus2: The tutorial suggests to restore checkpoints for loading a model. 
This is inconvenient, because you first need to process the data, build the 
whole architecture and then initialize it from the checkpoint. 
It would be much nicer to simply load a model and run translations with it.
 Can you find a way to achieve this?

Hand in all of your code, i.e. the working tutorial code along with all changes/
additions you made. Include outputs which document some of your experiments. Also
remember to answer the questions above! Of course you can also write about other
observations you made.