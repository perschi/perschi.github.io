---
layout: post
title:  "Information leakage in batch normalization"
date:   2021-02-23 08:03:48 +0100
categories: deep learning, batch normalization
---
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        processEscapes: true
      }
    });
    </script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Introduction
In the recently published paper [High-Performance Large-Scale Image Recognition Without Normalization][brock-21] Brock et al. mention as a motivation for the normalization free approach:

> "[...] batch normalization cannot be used for some tasks, since the interaction
> between training examples in a batch enables the network to
> ‘cheat’ certain loss functions. For example, batch normalization requires specific 
> care to prevent information leakage in
> some contrastive learning algorithms ([Chen et al., 2020][chen-20]; [He et al., 2020][he-20])"

This statement and the arguments in the references made me wonder, how severe this information leakage in batch normalization really is.
To investigate the leakage, I conduct some small experiments to find out what could be possible.

## The Inputs of Others

In the first experiment, I chose the arguably worst case scenario for batch norm with a batch size of two and try to train an
autoencoder that reconstructs input for the other sample on the MNIST data set. As the autoencoder I choose the following network:

```Python3
transform = nn.Sequential(
        nn.Conv2d(1, 64, 5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64,64, 5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64,64, 5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64,64, 5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * (28 - 4*4)**2, 10),
        nn.Linear(10, 64 * (28 - 4*4)**2), nn.ReLU(),
        ReshapeLayer([-1, 64,(28 - 4*4),(28 - 4*4)]),
        nn.ConvTranspose2d(64,64,5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.ConvTranspose2d(64,64,5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.ConvTranspose2d(64,64,5), nn.BatchNorm2d(64), nn.ReLU(),
        nn.ConvTranspose2d(64, 1,5))
```
This network is then used to train the following loss, which is the mean squared error between the output and the other input in the batch.

```Python3
F.mse_loss(transform(x),torch.flip(x, dims=[0]))
```

This networks solves the task surprisingly well. As you can see in the figure below the reconstruction of the image resembles the other input in training mode.
However, if the network is switched to test mode the reconstruction becomes arbtray but still resembles the original input.

<img src="/assets/information_leakage_in_batch_normalization/example0.svg">

To put this into numbers, I use pairs of the test set and calculate average loss over all batches in training mode and test mode.
Here, training mode achieves an average mean-squared error of 0.019 whereas test mode achieves only 0.098 which is almost an order of magnitude higher.
This also shows how signifcant the difference between the training and test estimates can be in the worst case.
However, it is known that batch normalization works better with larger batch size **(TODO add sources)**.
Unfortunately, this experiment does not scale trivially to larger batch sizes since this would require some kind of strong positional encoding of the batch element and target batch element.
This, is unlikely to happen in real scenarios. For that reason, I want to test performance in a more realistic scenario that should have similar problems as the
contrastive learning algorithms.
There, the issue is that the learning target would be contained in the the same batch allowing the information to cheat.
Hence, I want to build a data set showing the same issue by example.

## Hidden in Plain Batch

# The Data

To create a data set that contains that issue I use a simple function that should be not too easy to predict: 

$$ f(x) = sin(x) + 2 \cdot cos(0.5 \cdot x) $$ 

<img src="/assets/information_leakage_in_batch_normalization/function.svg">

This function is than sampled regularly at 1000 in the intervall $$[ 0, 2 \pi ]$$.
These values are then interpreted as a sequence of values where I take a small subsequence of the values to predict the value of the next one.
However, to introduce the problematic behavior each training instance consists of a batch of sequences where the values are shifted by one,
so that the solution to the element is in the actual batch.
For instance, consider the function values $$ ..., 1.2, 1.4, 1.2, 1.1, 0.9, ...$$ then one sequence batch with sequence_length 3 would be 

$$((1.2,1.4),1.2), ((1.4, 1.2), 1.1), ((1.2, 1.1),0.9)$$

This scenario is much more realistic than the previous example as it is similar to the contrastive learning approaches by [Chen et al., 2020][chen-20] and [He et al., 2020][he-20]. However, this situation could also arise more subtle if you work on raw audio generation. Here, you could split the audio into overlapping
sequences and with a bit of bad luck you could get overlapping sequences in the the same batch.

# The Experiment

As for the experiment with this data set, I train the following model for sequence lengths in $$\lbrace 4, 5, ..., 20 \rbrace $$ and batch sizes in $$\lbrace 1, 2, ..., 7 \rbrace $$.

```Python3
model = nn.Sequential(
    nn.Conv1d(1 , 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(),
    nn.Conv1d(32, 1, sequence_length - 1), nn.Flatten()).to(device)
```


In this case batch size refers to the number of sequence batches in a single batch. All models and combinations are then trained for 1000.0 steps.

# The Result

For the evaluation, I compare the average loss per sequence batch on the training data set with batch normalzation in training mode $$ L_{train} $$ and test mode $$ L_{test} $$.
In the case that the network is able cheat to use the information leakage in batch normalzation,  $$ L_{train} $$  is smaller than $$ L_{test} $$.
Otherwise, the network is forced to generalize.

<img src="/assets/information_leakage_in_batch_normalization/cheatfair.svg">

In this figure summarizing the results, we can see that the there is a connection between the batch size and the sequence length and the classification as a cheating network or a generalizing network. Namely, that the higher the sequence length, the more likely the network is to cheat even with larger batch sizes.
For that reason, we have another argument to use large batch sizes when applying batch normalization, when we do not want interaction between elements.
But, what if the information leakage could be used.

### 

[brock-21]: https://arxiv.org/abs/2102.06171
[chen-20]: https://arxiv.org/abs/2002.05709
[he-20]: https://arxiv.org/abs/1911.05722
