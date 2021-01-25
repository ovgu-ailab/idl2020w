---
layout: default
title: Assignment 12
id: ass12
---


# Assignment 12: Introspection Part 2
**Deadline: February 1st, 11am**  
This is the final assignment for this class.

In this assignment, you will try to detect misbehavior of models and explain errors using Introspection methods.

## Unmasking Clever Hans Predictors
We will start with a synthetic example, where we make it specifically easy for the model to cheat. Afterwards, we will apply Introspection to detect that and explain how the model cheated.

- start with a simple data set and augment the examples of one class with some kind of identifier, which can be
  - very obvious, like a common text (e.g. time stamp) added to the image - or a big black square
  - less obvious, like a highly transparent watermark
  - a particular image enhancement (like adding a lightness gradient from one corner to the other)
- train a model on this data set  
  it should perform very well on the augmented class, because of the added information
- finally, apply Introspection techniques on the trained model to figure out whether the model really made use of the cheating possibility that we provided  

Instead of only altering one class, you could also apply different kinds of cheating opportunities to different classes.

Hint: Your regularization should not get rid of the information that we have inserted into the class. Otherwise this experiment is useless.

## Contrastive Explanations
This is a more realistic scenario, in which you can also try out more advanced methods to create saliency maps.   
You will work with a pre-trained network and try to explain wrong decisions of the network with different Introspection techniques and contrastive explanations.  

- pick a pre-trained model of your choice (image classification would be best)
- gather examples that were wrongly predicted by the model, so you have a list of images with their wrong predicted label and their correct annotated label   
  (the examples do not need to come from the training data, but it's recommended that they are not too far away from the training data distribution)
- now compute saliency maps, e.g. using [tf-explain](https://tf-explain.readthedocs.io/en/latest/), for both the predicted class and the correct label
- compare the two saliency maps and investigate
  - are they different? (maybe you need to look at the actual difference of the saliency maps)
  - can you learn something about why the model made the error?   
    either from the saliency map of the predicted class only, or only by comparing to the target class saliency map

If you use tf-explain (or something similar), you can easily try out different saliency map methods and compare which one helps you most in explaining classification errors.

## Bonus: An experiment: explanation by input optimization
Let's use our feature visualization technique from the last assignment in a different way.

- train a model, e.g. on Cifar10, like last time
- pick examples that were wrongly classified, together with their predicted class and their annotated label.
- now optimize the example, such that it is classified as the correct class  
  We want to keep the changes as small as possible. Therefore, you need to add a penalty to the loss function to account for this (e.g. penalize pixel difference of optimized input to original input)
- finally inspect what the optimization needed to change in the image to make the model detect it as the correct class  (e.g. inspecting the difference image)
- Can you learn something about why the error was made from the changes?

Note from Andreas: I have no idea, whether this approach works. But I would be very excited to see some experiments :)
