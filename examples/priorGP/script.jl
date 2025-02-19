# # Approximating a GP Prior

# **Load required packages**
using KernelSpectralDensities
using AbstractGPs
using StatsBase
using CairoMakie
using DisplayAs #hide

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

# ## Approximating a GP
# One of the reasons to be interested in the spectral density of a kernel is
# that it allows us to approximate a GP Prior.