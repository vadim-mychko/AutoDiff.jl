# AutoDiff.jl

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

`AutoDiff` is a Julia package designed for automatic differentiation, enabling efficient and precise gradient computations necessary for a wide range of applications including machine learning, optimization, and numerical analysis.
Central to the package is the Tensor struct, which encapsulates not only the data but also the computational graph: a record of operations performed, allowing backward propagation of gradients.
