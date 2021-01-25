---
layout: default
title: Assignment 12
id: ass12
---


# Assignment 12: Introspection Part 2
**Deadline: February 1st, 11am**  
This is the final assignment for this class.

In this assignment, you will try to detect misbehavior of models and explain errors
 using Introspection methods.

## Unmasking Clever Hans Predictors
We will start with a synthetic example, where we purposefully make it easy for
 the model to cheat. Afterwards, we will apply Introspection to detect _that_
  and explain _how_ the model cheated.

- Start with a simple data set and augment the examples of one class with some
 kind of identifier, which can be
  - very obvious, like a common text (e.g. time stamp) added to the image, 
  or maybe a big black square,
  - less obvious, like a highly transparent watermark,
  - a particular image enhancement (like adding a lightness gradient from one
   corner to the other)...
- Train a model on this data set.
  It should perform very well on the augmented class because of the added
   information.
- Finally, apply Introspection techniques on the trained model to figure out 
whether the model really made use of the cheating possibility that we provided. 

Instead of only altering one class, you could also apply different kinds of
 cheating opportunities to different classes.

Note: If you add regularization to your model, this should not get rid of the
 information that we have inserted into the class. For example, if your "extra
 information" is that the images of one class are highly saturated, but you add
 preprocessing that normalizes image saturation, or perhaps applies random
 saturation to _all_ images, this would destroy the extra information.

## Contrastive Explanations
This is a more realistic scenario in which you can also try out more advanced
 methods to create saliency maps.   
You will work with a pre-trained network and try to explain wrong decisions of
the network with different Introspection techniques and contrastive explanations.  

- Pick a pre-trained model of your choice (image classification would be best).
- Gather examples that were wrongly predicted by the model, so you have a list 
of images with their wrong predicted label and their correct annotated label 
(the examples do not need to come from the training data, but it's recommended
 that they are not too far away from the training data distribution).
- Now compute saliency maps, e.g. using 
[tf-explain](https://tf-explain.readthedocs.io/en/latest/),
 for both the predicted class and the correct label.
- Compare the two saliency maps and investigate:
  - Are they different (maybe you need to look at the actual difference of the
   saliency maps)?
  - Can you learn something about why the model made the error (either from the
   saliency map of the predicted class only, or by comparing to the 
   target class saliency map)?

If you use tf-explain (or something similar), you can easily try out different
saliency map methods and compare which one helps you most in explaining 
classification errors.

## Bonus: An experiment: Explanation by input optimization
Let's use our feature visualization technique from the last assignment in a 
different way.

- Train a model, e.g. on Cifar10, like last time.
- Pick examples that were wrongly classified, together with their predicted
 class and their annotated label.
- Now optimize the example such that it is classified as the correct class. We
 want to keep the changes as small as possible. To this end, you can add a
  penalty to the loss function (e.g. penalize pixel
   difference of optimized input to original input).
- Finally inspect what the optimization needed to change in the image to make
 the model detect it as the correct class (e.g. by inspecting the difference
  image).
- Can you learn something about why the error was made from the changes in the
input?

Note from Andreas: I have no idea whether this approach works.
 But I would be very excited to see some experiments :)
