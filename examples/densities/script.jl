
# # Spectral Density of a Kernel
# All stationary kernels can have a spectral density, which is the Fourier transform 
# of the function $k(\tau) = k(x, x')$, where $\tau = t - t'$.
#
# In other words, the spectral density is defined as
### ToDo: Check with my definition. #src
# ```math
#   S(\omega) = \int_{-\infty}^{\infty} k(\tau) \exp(-i \omega \tau) d\tau
# ```
#
# In this notebook we show how we can recover the kernel from its spectral density.

# ## Load required packages
using KernelSpectralDensities
using Distributions
using LinearAlgebra
using FastGaussQuadrature
# using StatsBase ## not sure I need it in this notebook? #src

using CairoMakie

# ## Intro
# First we define some intervals
τ_interval = [0., 4.]
τv = range(τ_interval..., length=60)

w_interval = [-2.0, 2.0];

# Then we define a simple matern kernel with a lengthscale, 
# so that we don't keep it too simple.
### ToDo: Do multiple kernel here, for comparison #src
ker = Matern32Kernel()
l = 0.5
ker = with_lengthscale(ker, l)

# Now we can use a function from KernelSpectralDensities.jl to 
# get its spectral density.
# The resulting object allows us to evaluate the spectral density for any frequency. 
S = SpectralDensity(ker, 1)
S(0.2)

# We can also plot it over the interval we defined to see its shape.
w_plot = range(w_interval..., length=151)
lines(w_plot, S.(w_plot), label="spectral density")

# ## Recovering the kernel
# We can recover the kernel by integrating the spectral density over all frequencies.
# 
# First, we we define the stationary function and some interals
k(t) = ker(0, t)

# For the numerical integration we use the GaussLegendre quadrature schema, 
# which is more accurate and efficient than equidistant intervals.  
# This allows us to define a new function, which numerically approximates 
# the inverse Fourier transform of the spectral density.

wv, weights = gausslegendre(500)
wv = (wv .+ 1) ./ 2 * (w_interval[2] - w_interval[1]) .+ w_interval[1]
c = (w_interval[2] - w_interval[1]) / 2

ks(t) = c * sum(S.(wv) .* cos.(2 * π * wv * t) .* weights)

# ## Results
# We see that we indeed recover the kernel from the spectral density, 
# with only a small error from the numerical integration.
f = Figure()
ax = Axis(f[1, 1])
lines!(ax, τv, k.(τv), label="kernel")
lines!(ax, τv, ks.(τv), label="spectral approx")
axislegend()
f


