 [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 

# NN-Without-Frameworks
This project aims to implement different Neural Network configuration **without** using scientific frameworks like **TensorFlow** or **PyTorch**.

Each network/config is implemented in 4 formats while trying to mimic [PyTorch's Subclassing](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html) procedure:
1. In Python using NumPy
2. In Python without taking advantage of NumPy
3. In java
4. in C++

## Examples
> currently only regression is supported.

Each directory contains a `train_regression.*` that performs test of correctness and functionality according to its corresponding format. You can execute them to get a sense of what is going on.

## Acknowledgement 
Current code is inspired by the elegant and simple repository [Simple Neural Networks](https://github.com/MorvanZhou/simple-neural-networks) by [@MorvanZhou ](https://github.com/MorvanZhou).

## Contributing
- The current code is far from done and any fix, suggestion, pull requests, issues, etc is highly appreciated and welcome. 🤗
- Current work focuses on discovering what is under the hood rather optimization involved in implementing ideas so, feel free to conduct sanity-checks behind the math and correctness of each part and/or if you come up with a better or optimized solution, please don't hesitate to bring up a PR. [thank you in advance. 😊]
- You can take a look at `todo.md`.

