⚡ Engineered by Kiliandiama | The Diama Protocol [10/10] | All rights reserved.
OjaNeuronLayer_AVX512

Ultra-Optimized Oja Neuron Layer for C++ (AVX-512 + OpenMP)

This repository contains a high-performance implementation of an Oja neuron layer, optimized for modern CPUs using AVX-512 vectorization and OpenMP parallelization. It is designed for fast online learning and neural simulations.

Features

Fully vectorized with AVX-512 for 8 doubles per vector.

Optional OpenMP parallelization for multi-core CPUs.

RAII-managed 64-byte aligned arrays for maximum SIMD efficiency.

Implements Oja's learning rule for unsupervised neural learning.

Portable and safe modern C++17 code.
