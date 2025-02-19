# # Random Fourier Features
# One of the reasons to be interested in the spectral density of a kernel is 
# that it allows us to approximate a GP Prior.
# 
# In this notebook we show the two feature functions implemented in KernelSpectralDensities.jl 
# and how to use them.

# **Load required packages**
using KernelSpectralDensities
using StatsBase
using CairoMakie
using DisplayAs #hide

# ## Intro
# In general, feature functions allow us to project an input into a higher-dimensional space,
# which is useful for a variety of tasks.
# For example, we can use them to approximate a kernel using the "kernel trick".

# A special class of feature functions are "random Fourier features", derived from 
# the Fourier transform, which we saw in add link from other example. 
# KernelSpectralDensities implements two types of random Fourier features, 
# `ShiftedRFF` and `DoubleRFF`.
# 
# For this example we use the simple squared exponential kernel.
ker = SqExponentialKernel()
S = SpectralDensity(ker, 1)

# ## ShiftedRFF
# The `ShiftedRFF` feature function is somewhat more common, and has 
# been used in papers such as [Efficiently sampling functions from Gaussian process posteriors](https://proceedings.mlr.press/v119/wilson20a.html).
#
# It is defined as
# ```math
#     \varphi(x) = \sqrt{2 / l}  \cos(2  π  ((w^T  x) + b))
# ```
# where `w` is sampled from the spectral density `S`, 
# `b` is uniformly sampled from `[0, 2π]` 
# and `l` is the number of sampled frequencies, which is also 
# the number of features.
#
# We generate a set of 4 feature functions, which we can evaluate 
# at any point.
srff = ShiftedRFF(S, 4)
srff(1.0)

# If we plot them, we see that each feature function is a harmonic 
# with varying frequency and phase.
x = range(0, 5; length=100)
f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="x", ylabel="rff(x)", title="ShiftedRFF")
series!(ax, x, reduce(hcat, srff.(x)); labels=["srff $i" for i in 1:4])
axislegend(ax; position=:rt)
f
DisplayAs.PNG(f) #hide #md

# ## DoubleRFF
# The `DoubleRFF` feature function is less common, but is theoretically
# equivalent to the `ShiftedRFF` feature function.
#
# It is defined as
# ```math
#     \varphi(x) = \sqrt{1 / l} \begin{pmatrix} \cos(2 π w' x) & \sin(2 π w' x) \end{pmatrix} 
# ```
# where `w` is sampled from the spectral density `S`,
# with a total of `l/2` sampled frequencies.
#
# Here, each function is effectively two feature functions in one, 
# so specifying `l` will result in `l/2` samples but an `l` dimensional 
# feature vector.
# 
# We again generate a set of 4 feature functions. 
drff = DoubleRFF(S, 4)
drff(1.0)

# We plot these features as well
f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="x", ylabel="rff(x)", title="Double RFF")
series!(ax, x, reduce(hcat, drff.(x)); labels=["drff $i" for i in 1:4])
axislegend(ax; position=:rt)
f
DisplayAs.PNG(f) #hide #md

# ## Approximating a kernel
# Tbd