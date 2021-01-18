---
layout: default
title: Assignment 11
id: ass11
---


# Assignment 11: Introspection Part 1
**Deadline: January 25th, 11am**


In this assignment, you will implement gradient-based model analysis both for
 creating saliency maps (local) and for feature visualization (global). 
 You can adapt your implementation of the adversarial examples of Assignment 10 and also take inspiration from the DeepDream tutorial.

You can use any CIFAR10 or MNIST Model, e.g. from the tasks before.

## Gradient-based saliency map (sensitivity analysis)

Run a batch of inputs through the trained model.
Wrap this in a GradientTape where you watch the input batch
(batch size can be 1 if you'd like to just produce a single saliency map).
and compute the gradient for a particular logit or its softmax output _with 
respect to the input_.
This tells us how a change in each input pixel would affect the class output.
This already gives you a batch of gradient-based saliency maps!
Plot the saliency map next to the original image or superimpose it.
Do the saliency maps seem to make sense? How would you interpret them?

Hint: It makes sense to take the sign of the gradient into account when 
interpreting them.
Negative gradients indicate a decrease in output value, positive 
gradients an increase.

## Activation Maximization
Extend the code from the previous part to create an optimal input for a 
particular class.

Multiply the gradients with a small constant (like a learning rate) and add them
to the input.
Repeat this multiple times, computing new gradients with respect to the input each
time.
Essentially, you are writing a "training loop" for producing an optimal input for
a certain class (do _not_ train the model weights!).

Does the resulting input look natural?
How does the inputs change when applying many steps of optimization?
How do the optimal inputs differ when initializing the optimization with random 
noise instead of real examples?
Can you see differences between optimizing a logit or a softmax probability?

**Bonus**: Apply regularization strategies to make the optimal input more 
natural-looking.
You can also optimize for _hidden features_ of the network (instead of outputs)
assuming you can "extract" them from the model you built.
