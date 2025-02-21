# # Sampling a GP Prior

# **Load required packages**
using KernelSpectralDensities
using AbstractGPs
using StatsBase
using LinearAlgebra
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
# In this example, we want to sample the GP at the following points
x_sample = range(0, 2; length=5)

# To sample, we calculate the mean and covariance of the GP at these points. 
# While we use the AbstractGPs interface, in this case the mean is just
# a zero vector and the covariance is the kernel matrix over the sample points.
m = mean(gp, x_sample)'
#-
K = cov(gp, x_sample)
# 
# We then compute the Cholesky decomposition of the covariance matrix
# samples a vector of standard normal random variables and obtain a 
# sample from the GP prior.

function naive_sample(gp, x_sample)
    m = mean(gp, x_sample)
    K = cov(gp, x_sample)
    Kc = cholesky(K).L
    ζ = randn(length(x_sample))
    return m .+ Kc * ζ
end

ys = naive_sample(gp, x_sample)
# 
# To illustrate we plot a few samples
x_plot = range(0, 1; length=10)
n_samples = 7
ys_plot = [naive_sample(gp, x_plot) for _ in 1:n_samples]

f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="x", ylabel="y", title="Naive Sampling")
series!(ax, x_plot, reduce(hcat, ys_plot)'; labels=["sample $i" for i in 1:n_samples])
axislegend(ax; position=:rt)
f
DisplayAs.PNG(f) #hide #md

# To evaluate the samples, we define the following function
function evaluate_samples(y_sample, m, K)
    ms = mean(y_sample)
    merr = norm(m .- ms)
    cs = cov(y_sample)
    cerr = norm(K .- cs)
    println("Mean error: $merr, Covariance error: $cerr\n")
    println(ms)
    return cs
end
# For the small number of samples we have, the results are not very good. 
y_sample = [naive_sample(gp, x_sample) for _ in 1:n_samples]
evaluate_samples(y_sample, m, K)

#
# If we sample a lot more functions however, we get closer to the anaytical result
n_manysamples = 1000
y_sample = [naive_sample(gp, x_sample) for _ in 1:n_manysamples]
evaluate_samples(y_sample, m, K)

#
# However, there are two issues with this approach: 
# 1. It is quite computationally expensive, since we need to calculate the Cholesky decomposition.
# 2. Sampling at a larger number of points can cause conditionint issues, as we show below.   
x_sample_many = range(0, 2; length=20)
try
    naive_sample(gp, x_sample_many)
catch err
    showerror(stderr, err)
end

# ## RFF Sampling
# Random Fourier features are an alternative option to sample the GP prior.
# Instead of computing the Cholesky decomposition of the kernel matrix, we
# compute a number of Fourier features and can generate samples from the GP
# by defining a weighted sum of these features. 
# ```math
#     f(x) = \sum_{i=1}^l w_i \varphi_i(x)
# ```
# The weights $w_i$ are sampled from a standard normal distribution.

rff = DoubleRFF(S, 10)
agps = ApproximateGPSample(rff)
agps.(x_sample)

# We can plot the samples as before
n_samples = 7
ys_plot = [ApproximateGPSample(rff).(x_plot) for _ in 1:n_samples]

f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="x", ylabel="y", title="RFF Sampling")
series!(ax, x_plot, reduce(hcat, ys_plot)'; labels=["sample $i" for i in 1:n_samples])
axislegend(ax; position=:rt)
f
DisplayAs.PNG(f) #hide #md

# Unfortunately, the mean and the covariance are worse than with the naive sampling
# for the same number of samples. 
y_sample = [ApproximateGPSample(rff).(x_sample) for _ in 1:n_samples]
evaluate_samples(y_sample, m, K)

# However, we now have another parameter to tune: The number of features
# By increasing the number of features, we get close to the result we saw 
# with the naive sampling.
rff500 = DoubleRFF(S, 500)
y_sample = [ApproximateGPSample(rff500).(x_sample) for _ in 1:n_samples]
evaluate_samples(y_sample, m, K)

# By increasing the number of GP samples, we can again improve the results in 
# both cases. 
# 
# With 10 feature functions
y_sample = [ApproximateGPSample(rff).(x_sample) for _ in 1:n_manysamples]
evaluate_samples(y_sample, m, K)
#
# With 500 feature functions
y_sample = [ApproximateGPSample(rff500).(x_sample) for _ in 1:n_manysamples]
evaluate_samples(y_sample, m, K)

#
# Lastly, we note that we no longer have to worry about conditioning issues,
# and can evaluate a given GP sample at however many points we like
ApproximateGPSample(rff).(x_sample_many)
