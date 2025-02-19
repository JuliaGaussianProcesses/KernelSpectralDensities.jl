# # Random Fourier Features
# One of the reasons to be interested in the spectral density of a kernel is 
# that it allows us to approximate a GP Prior.
# 
# In this notebook we show the two feature functions implemented in KernelSpectralDensities.jl 
# and how to use them.

# ## Load required packages
using KernelSpectralDensities
using AbstractGPs
using StatsBase
# using LinearAlgebra
using AbstractGPsMakie
using CairoMakie

# ## Intro
# We use the AbstractGPs package to define a stationary GP prior,
# in other words, a GP that has not been conditioned on data yet.

ker = SqExponentialKernel()
S = SpectralDensity(ker, 1)

gp = GP(ker)

# We can also plot this GP using AbstractGPsMakie.jl, but 
# we don't see very much, since we have a simple GP with 
# zero mean and a variance of 1. 
f = plot(0:0.1:1, gp; size=(600, 400))
DisplayAs.PNG(f) #hide #md

# ## Random Fourier Features
# KernelSpectralDensities implements two types of random Fourier features, 
# `ShiftedRFF` and `DoubleRFF`.
# A feature function projects its input into a higher-dimensional "features" space.,

# ### ShiftedRFF
# The `ShiftedRFF` feature function is somewhat more common, and has 
# been used in papers such as [Efficiently sampling functions from Gaussian process posteriors](https://proceedings.mlr.press/v119/wilson20a.html).
#
# It is defined as
# ```math
#     \sqrt{2 / l}  \cos(2  π  ((w^T  x) + b))
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
axislegend(ax; position=:ct)
f
DisplayAs.PNG(f) #hide #md

# ### DoubleRFF
# The `DoubleRFF` feature function is less common, but is theoretically
# equivalent to the `ShiftedRFF` feature function.
#
# It is defined as
# ```math
#     \sqrt{1 / l} [\cos(2 π w' x),  \sin(2 π w' x)]
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
axislegend(ax; position=:ct)
f
DisplayAs.PNG(f) #hide #md