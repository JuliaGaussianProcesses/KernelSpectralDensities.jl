# # Sampling a GP Prior

# **Load required packages**
using KernelSpectralDensities
using AbstractGPs
using StatsBase
using AbstractGPsMakie
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

# ## Naive Sampling
# If we want to draw a sample from the GP prior, the 
# standard way is to use the Cholesky decomposition of the kernel matrix.
#
# In this example, we want to sample the GP as the following points
x = range(0, 1, length=5)
y = 

# ## RFF Sampling
# We can also use the feature functions to draw approximate samples 
# from a GP prior. 
