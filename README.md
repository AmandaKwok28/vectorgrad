# VectorGrad

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)
![NumPy](https://img.shields.io/badge/dependency-NumPy-orange)
![Inspired by](https://img.shields.io/badge/inspired%20by-micrograd-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)


## Highlights

- Educational implementation (not optimized for production use)
- Vectorized reverse-mode autodiff engine
- NumPy-backed tensor operations
- Explicit broadcast / unbroadcast gradient handling
- Graph-based backpropagation implementation
- Minimal, inspectable core designed for learning


## Overview

This repository contains a minimal vectorized automatic differentiation engine built as a learning project. I created it to better understand how gradients, computation graphs, and broadcasting work in modern ML systems by implementing them explicitly in NumPy. Inspired by micrograd, the code prioritizes readability and transparency over performance, making it easy to inspect, modify, and experiment with.


## Installation
Creating a conda environemnt: (optional but recommended)

```bash
conda create -n vectorgrad python=3.10
conda activate vectorgrad
```


Clone the repository:
```bash
git clone https://github.com/AmandaKwok28/vectorgrad
cd vectorGrad
```

Install minimal dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

To explore vectorgrad on a toy XOR dataset, navigate to the `/demo` directory and run the vectorgrad.ipynb notebook.
The notebook walks through basic usage, forward passes, and gradient computation using vectorized tensors.

To import the core `Tensor` class in your own code:

```py
>>> from vectorgrad.engine import Tensor
```


## Testing 
From the root directory of the project, run:

```py
python -m vectorgrad test
```

The above runs all tests in the test folder. I'm still updating them as I learn. Will develop this a lot more once all operations are batched.


## Intuition

In the `/docs` folder I’m slowly adding the math behind each operation. It’s coming along, but typing LaTeX takes me forever — sorry! I’ve found that once you derive the math, most of it is fairly straightforward. The main conceptual hurdles are unbroadcasting and the backward() logic itself; everything else reduces to basic calculus and linear algebra.


## Note:

I'm working on the LeNet demo right now. My operations aren't batched for avg_pool2d, flatten, and softmax so I need to fix that in order for it not to take hours to process the MNIST digits dataset. Should be out soon though! 

Also, I know that conv2d batching can be a lot faster if I made calls to einsum or BLAS but I'm keeping it transparent to expose the underlying math for each method. Shouldn't take 
that long for toy examples like MNIST (hopefully)

The current MLP demo is fully functional and serves as the primary example for now. It's listed as `vectorgrad.ipynb` in the `/demo` folder
