---
layout: post
title:  "Layer Normalization is a hyperspherical projection"
date:   2021-03-09 00:00:0 +0000
categories: deep learning, Layer Normalization
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

# Introduction

During my master thesis, I needed to stabilize the training of my deep neural network. The typical decision in literature is to use Batch Normalization.
However, the available GPU memory was limited, so that it was not reasonable to apply due to the small possible batch size.
Instead, I used [Layer Normalization][Ba-16-Layernorm] which is not dependent on the batch size.
Here, the normalization is applied over the elements in the batch instead of the batch.

Now, I wondered what does Layer Normalization to the input data. For that, let's have a look at the results of Layer Normalization in 2D.

<img style="width:50%"  src="/assets/layer_normalization_is_a_hyperspherical_projection/Input_data.svg"><img  style="width:50%" src="/assets/layer_normalization_is_a_hyperspherical_projection/output_data.svg">

For that, I randomly sampled the two-dimensional data points on the left and applied Layer Normalization which results in the right plot.
These resulting data points of the image are surprising since almost all data points are projected onto two points.
Maybe, we can see what is happening if we move a dimension higher and repeat the experiment with 3D points.
<img src="/assets/layer_normalization_is_a_hyperspherical_projection/input3d.svg"><img src="/assets/layer_normalization_is_a_hyperspherical_projection/output3d.svg">


As we see, the data points are projected on a sphere in a dimension lower than the data. After discovering this, I wondered why this is and looked up
if anyone has discussed this behavior. However, the only mentioning of this hyperspherical projection I found was this [reddit post][reddit-1].
Before going into discussing this property let's take a look at why Layer Normalization is a hyperspherical projection.

# The Math behind 

Here, we start by looking at how Layer Normalization works and then try to prove that all points interpreted as vectors are in the same hyperplane with
the one-vector as normal-vector.
In the next step, we prove it is a sphere by calculating the norm of instances that is only dependent on dimensionality.

## Calculating Layer Normalization

In Layer Normalization, we consider an input vector $$x = (x_1, ..., x_n)$$. Then the normalized vector is $$\bar{x} = \frac{x - \mu}{\sigma}$$.
Here, $$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$ and $$\sigma = \sqrt{\frac{1}{n} \sum_{i=-1}^n (x_i - \mu)^2}$$.
Furthermore, we assume $$\exists_{i,j} : x_i \neq x_j$$ so that we do not have to worry that $\sigma = 0$ in the denominator.
If this case is present, we get the artifacts we see in the 2d normalization where one point is not exactly on the two points.
The reason for that is the addition of an $$\varepsilon$$ in the denominator that is added to circumvent dividing by zero.

## Property 1: Orthogonality to the one-vector
From the two plots, I assumed the orthogonality to the one-vector.
To show this, we just have to determine that scalar product and as you can see below it is orthogonal to the one vector.

$$ <\bar{x}, \mathbb{1}> = \frac{(x_1 - \frac{1}{n}\sum_{i=1}^{n} x_i ) + ... + (x_n - \frac{1}{n}\sum_{i=1}^{n} x_i)}{\sigma} = \frac{\sum_{i=1}^{n} x_i - \sum_{i=1}^{n} x_i}{\sigma} = 0$$

This means we know that all normalized instances can be found in the same hyperplane.

## Property 2: Normalized norm

Next, we look at the norm and calculate for a general instance.

$$||\bar{x}|| = \sqrt{\frac{(x_1 - \mu)^2}{\sqrt{\frac{1}{n} \sum_{i=-1}^n (x_i - \mu)^2}^2} + ... + \frac{(x_n - \mu)^2}{\sqrt{\frac{1}{n} \sum_{i=-1}^n (x_i - \mu)^2}^2} }$$
$$= \sqrt{\frac{\sum_{i=1}^{n} (x_i - \mu)^2}{\frac{1}{n} \sum_{i=-1}^n (x_i - \mu)^2}} = \sqrt{\frac{1}{\frac{1}{n}}} =  \sqrt{n} $$

This result means that all normalized data points are in the same hyperplane and have the same distance to the origin which defines a hypersphere.

# Discussion

Let's have look at some implications of this finding. For instance, we can look at [Group Normalization](Wu-18-groupnorm) which can be seen as a
generalization where [Layer Normalization](Ba-16-Layernorm) and [Instance Normalization](Ulyanov-16) are special cases.
The general idea is similar to Layer Normalization, but instead of normalizing over the complete vector, the vector is partitioned into
$$G$$ same-sized groups.

## Group Normalization
Let's have a look if the shown properties from Layer Normalization still hold for Group Normalization.

### Group Normalization: Orthogonality to the one vector

To formalize the task, we enumerate the elements of the input vectore as follows where the first index is index the group index and
the second the element in group index.
$$ x = (x_{1,1}, x_{1,2}, ..., x_{1,k}, x_{2,1}, ... x_{2,k}, ..., x_{G,k}) = (x^{(1)}, ..., x^{(G)})$$

If we now just calculate everything as before, we see that the resulting normalized vector is still orthogonal to the one-vector.

$$ <(\bar{x}^{(1)}, ..., \bar{x}^{(G)}), \mathbb{1}> = \sum_{g} \sum_{k} \bar{x}_{g,k} = \sum_{g} <\bar{x}^{(g)}, 1> = 0$$

For this, we exploit our knowledge from the Layer Normalization calculation. From that, we know that each group by itself is orthogonal to the partial
one-vector. Therefore, we sum zeros and end at zero.

### Group Normalization: Normalized norm

$$|| (\bar{x}^{(1)}, ..., \bar{x}^{(G)}) || = 
\sqrt{ \frac{(x_{1,1} - \mu_1)^2}{\sigma_1^2} + ... + \frac{(x_{G,1} - \mu_G)^2}{\sigma_G^2} }=
\sqrt{ ||\bar{x}^{(1)}||^2 + ... + ||\bar{x}^{(G)}||^2} $$

$$= \sqrt{G \cdot \sqrt{k}^2} = \sqrt{k \cdot G}$$

For calculating the norm, we again use our observations from Layer Normalization, so that the elements can be summarized group-wise and then replaced
by $$\sqrt{k}$$ times the number of groups.

This means all these normalization techniques project to a hypersphere. However, usually, these methods employ dimension-wise scaling factors after the normalization which make a difference in the output.
Next, we want to look at how these scaling factors impact the hyperspherical projection.

## What do the scaling factors do?
The scaling factors in Layer Normalization are element-wise multiplication with a weight and addition of a bias.
The influence of the bias is simply the offset to the origin of the projected sphere. In contrast to that, the scaling is more interesting
since it allows the application of [affine transformations][affine-transformation] to the hypersphere.
This means, we can scale, reflect, rotate and shear the the hypersphere. To determine the specific factor in the transformation, we can make use of our
normal-vector and imagine only transforming it. For instance, if we want to rotate the 1D-hypersphere, we can use the following as scaling factors:

$$ \begin{pmatrix} 1& 1 \end{pmatrix} \cdot \begin{pmatrix} \cos(\theta)& - \sin(\theta)\\ \sin(\theta) & \cos(\theta) \end{pmatrix} = \begin{pmatrix} \cos(\theta) +\sin(\theta) & \cos(\theta) -\sin(\theta) \end{pmatrix}$$

If we use the resulting scaling factors the different 1D-hyperspheres for different $$\theta$$ can be seen below.

<img src="/assets/layer_normalization_is_a_hyperspherical_projection/rotation.svg">

Here you see that the sphere can be rotated by using these scaling factors. With this in mind, we should also be able to recreate the transformation matrix for scaling factors given randomly sampled normalized points. This can make these transformations more understandable.

A different question is where the normalized instance are projected relative to each other. This is the question we will be looking at next.

## Where are my normalized Points? 
<img src="/assets/layer_normalization_is_a_hyperspherical_projection/input3d_bn.svg"><img src="/assets/layer_normalization_is_a_hyperspherical_projection/output3d_bn.svg">

So, if we compare Layer Normalization to [Batch Normalization][ioffe-15] above, we can see that Batch Normalization scales the instances so that they are closer to a normal distribution. During that, it can project its instance theoretically into arbitrary positions
in its output space. In contrast, Layer Normalization projects only onto a transformed hypersphere that uses "less" of the available space.
This could lead to collapsing instances at the same position. This happens trivially if we compare a vector with a scaled one.
However, lets consider different vectors and different rotation matrices to rotate the vectors and compare the distances after the normalization.
<table style="width:100%">
 <tr>
    <th> Vector </th>
    <th>Yaw</th>
    <th>Pitch</th>
    <th>Roll</th>
  </tr>
  <tr>
    <td></td>
    <td>$$ \begin{pmatrix} 
    \cos(\theta) & 0 & -\sin(\theta) \\
    0 & 1 & 0 \\
    \sin(\theta) & 0 & \cos(\theta)
    \end{pmatrix}$$</td>
    <td>
    $$\begin{pmatrix} 
    1 & 0 & 0 \\
    0 & \cos(\theta)  &  - \sin(\theta) \\
    0 & \sin(\theta)  & \cos(\theta)
    \end{pmatrix}$$</td>
    <td>$$\begin{pmatrix} 
    \cos(\theta)  &  - \sin(\theta) & 0 \\
    \sin(\theta)  & \cos(\theta) & 0\\
    0 & 0  & 1

    \end{pmatrix}$$</td>
  </tr>
  <tr>
    <td>$$\begin{pmatrix} 1\\ 1 \\ 1 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_1_1_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_1_1_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_1_1_1.svg"></td>
  </tr>
  <tr>
    <td>$$\begin{pmatrix} 1\\ 1 \\ 0 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_1_1_0.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_1_1_0.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_1_1_0.svg"></td>
  </tr>
  <tr>
    <td>$$\begin{pmatrix} 1\\ 0 \\ 1 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_1_0_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_1_0_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_1_0_1.svg"></td>
  </tr>
  <tr>
    <td>$$\begin{pmatrix} 0\\ 1 \\ 1 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_0_1_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_0_1_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_0_1_1.svg"></td>
  </tr>
    <tr>
    <td>$$\begin{pmatrix} 1\\ 0 \\ 0 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_1_0_0.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_1_0_0.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_1_0_0.svg"></td>
  </tr>
    <tr>
    <td>$$\begin{pmatrix} 0\\ 1 \\ 0 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_0_1_0.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_0_1_0.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_0_1_0.svg"></td>
  </tr>
    <tr>
    <td>$$\begin{pmatrix} 0\\ 0 \\ 1 \\\end{pmatrix}$$</td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_yaw_0_0_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_pitch_0_0_1.svg"></td>
    <td><img src="/assets/layer_normalization_is_a_hyperspherical_projection/diff_roll_0_0_1.svg"></td>
  </tr>
</table>

As you can see in the table there is some unexpected behavior present.
For instance, we see that it matters in which dimension and direction we rotate since there is asymmetric behavior for many inputs and rotations.
But, let's go through the table one by one. First, there is the one-vector. In this case, we have the wanted behavior that there is only a peak with a small distance around the identity of the vector.
In the next three rows, we have two ones and a zero in different permutations. Here, there is a rotation that results in a symmetric difference regarding the rotation angle. However, in the other two cases, the difference is not symmetric with the rotation. This means Layer Normalization produces projections that do behave
differently depending on the kind of difference between vectors. This introduces another significant non-linearity in comparison to Batch Normalization.
Similarly, we see the unit-vector plots. The difference here is that some rotations do not change the vector.
However, these non-linearities seem to be specific to zero inputs. Hence, keep in mind that the ReLU activation can produce lots of zeros, so this is something that can happen.

To further look into that, we can compare how the differences between a reference point and a sampled ball around it are distributed after normalization.
For that, we can look at the following two figures for Batch Normalization and Layer Normalization.

<img style="width:50%" src="/assets/layer_normalization_is_a_hyperspherical_projection/dist_ln.svg"><img style="width:50%"  src="/assets/layer_normalization_is_a_hyperspherical_projection/dist_bn.svg">

Here, we can see that the distance after Batch Normalization is increasing with the radius of the ball around the reference point.
In contrast, after Layer Normalization, there are multiple modes of difference. This shows the same issue as the previous charts but in a less constructed way.


# Conclusion

In this post, we saw that Layer Normalization is a hyperspherical projection that is transformed by an affine transformation. Furthermore, we saw that the Group Normalization family exhibits the same properties when we ignore the scaling factors and biases. Finally, we found out that the type of differences in the inputs to Layer Normalization influences how different they are afterwards.

*The Code to reproduce the charts can be found [here][code]*

[ioffe-15]:https://arxiv.org/abs/1502.03167
[affine-transformation]:https://en.wikipedia.org/wiki/Affine_transformation
[Ulyanov-16]:https://arxiv.org/abs/1607.08022
[Wu-18-groupnorm]:https://arxiv.org/abs/1803.08494
[Ba-16-Layernorm]:https://arxiv.org/abs/1607.06450
[reddit-1]:https://www.reddit.com/r/MachineLearning/comments/4u0a74/160706450_layer_normalization/d5mo9jt?utm_source=share&utm_medium=web2x&context=3
[code]:https://github.com/perschi/deep-learning-blog-code