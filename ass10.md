---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: What's Wrong With Our Data?
**Deadline: January 18th, 11am**


[Following this link](https://drive.google.com/open?id=1s_YZsHfMAU7mTWHV2o40ni2Oc36vCL81),
 you will find a selection of CIFAR10 datasets. The archive contains
triplets of training, validation and test sets. For each, train a model on the
training set, making sure it "works" by checking its performance on the validation
set. When you're satisfied (feel free to fine-tune your models to achieve
as good of a validation set performance as possible), check performance on the test set.
Does it still work? That is, is the performance on the test set close to that on
the validation set? You should find that it is usually either significantly worse
(which makes us sad) or too good (which is suspicious).
For each triplet, find out (e.g. through inspection or 
computing statistics of the dataset) what's going wrong. Typical things to watch
out for include:

- Do the labels match the examples
(of course you cannot check this for each example)?
- Do the train, validation and test sets follow the same distribution?
- Are the different subsets disjunct?
- Are the sets balanced?
- Where the sets processed in the same way?
- etc.

If your unfamiliar with `.npz` files, see 
[here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html)
for reference.

Note:
- Make sure you train your models long enough, else the problems in the dataset
may not be revealed.
- You **don't** need to resolve the issues you find (but feel free to propose
how you _could_ do it).

Next, each of the problems in the data sets is artificially constructed and somewhat
exaggerated.
Think about which possible _real-world_ problems of data sets each example represents
and provide examples.

## Bonus: Practical Methodology

If you have spare time, you can try to put some of the insights you've gained from
the lecture on "practical methodology" into practice. You can use a dataset such
as CIFAR as a testing ground for your models. Some possibilities:

- Do a grid or random search for good hyperparameters of your model and/or
training procedure.
- Use data augmentation to improve generalization.
- Conduct error analyses to guide the search for possible improvements of your
models.

See whether, by following somewhat principled procedures, you have an easier time
building high-performing models compared to "just trying stuff".
It’s always a good idea to document your stepwise changes when improving your 
model in such a structured way.
This allows you to easily revert changes. Also, when coming back to the code 
after some time, you would otherwise have a hard time remembering which changes
were successful.



