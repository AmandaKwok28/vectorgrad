# VectorGrad

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)
![NumPy](https://img.shields.io/badge/dependency-NumPy-orange)
![Inspired by](https://img.shields.io/badge/inspired%20by-micrograd-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)


## Highlights

- Vectorized reverse-mode autodiff engine
- NumPy-backed tensor operations
- Explicit broadcast / unbroadcast gradient handling
- Graph-based backpropagation implementation
- Minimal, inspectable core designed for learning


## Overview

This repository contains a minimal vectorized automatic differentiation engine built as a learning project. I created it to better understand how gradients, computation graphs, and broadcasting work in modern ML systems by implementing them explicitly in NumPy. Inspired by micrograd, the code prioritizes readability and transparency over performance, making it easy to inspect, modify, and experiment with.


## Usage

To explore vectorgrad on the MNIST handwritten digits dataset, navigate to the `/demo` directory and run the vectorgrad.ipynb notebook.
The notebook walks through basic usage, forward passes, and gradient computation using vectorized tensors.

To import the core `Tensor` class in your own code:

```py
>>> from vectorgrad.engine import Tensor
```


## Installation
Clone the repository:
```bash
git clone https://github.com/AmandaKwok28/vectorGrad
cd vectorGrad
```

Install minimal dependencies:
```bash
pip install -r requirements.txt
```
