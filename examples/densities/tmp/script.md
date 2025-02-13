```@meta
EditURL = "../script.jl"
```

# Spectral Densities
All stationary kernels can have a spectral density, which is the Fourier transform
of the function $k(\tau) = k(x, x')$, where $\tau = t - t'$.

In other words, the spectral density is defined as
```math
  S(\omega) = \int_{-\infty}^{\infty} k(\tau) \exp(-i \omega \tau) d\tau
```

In this notebook we show how we can recover the kernel from its spectral density.

## Load required packages

````@example script
using KernelSpectralDensities
using Distributions
using LinearAlgebra
using FastGaussQuadrature
using OrderedCollections

using CairoMakie
using DisplayAs
````

## Intro
First we define a few kernels, from KernelFunctions.jl,
which is re-exported by KernelSpectralDensities.

````@example script
kers = OrderedDict(
    "Matern3/2" => Matern32Kernel(),
    "Matern3/2 0.8" => with_lengthscale(Matern32Kernel(), 0.8),
    "Matern3/2 1.2" => with_lengthscale(Matern32Kernel(), 1.2),
);
nothing #hide
````

We plot them here for illustration.

````@example script
τ_interval = [0.0, 4.0]
τv = range(τ_interval...; length=60)

f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="τ", ylabel="k(τ)")
for (key, ker) in kers
    lines!(ax, τv, ker.(0, τv); label=key)
end
axislegend()
f
DisplayAs.PNG(f) # hide
````

Now we can use a function from KernelSpectralDensities.jl to
get its spectral density.
The resulting object allows us to evaluate the spectral density for any frequency.

````@example script
S = SpectralDensity(kers["Matern3/2"], 1)
S(0.5)
````

We can also plot it over the interval we defined to see its shape.

````@example script
w_plot = range(-1, 1; length=151)

f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="ω", ylabel="S(ω)")
for (key, ker) in kers
    Sp = SpectralDensity(ker, 1)
    lines!(ax, w_plot, Sp.(w_plot); label=key)
end
axislegend()
f
DisplayAs.PNG(f) # hide
````

## Recovering the kernel
We can recover the kernel by integrating the spectral density over all frequencies.

First, we we define the stationary function and some interals

````@example script
ker = kers["Matern3/2"]
k(t) = ker(0, t);
nothing #hide
````

For the numerical integration we use the GaussLegendre quadrature schema,
which is more accurate and efficient than equidistant intervals.
This allows us to define a new function, which numerically approximates
the inverse Fourier transform of the spectral density.

````@example script
w_interval = [-2.0, 2.0]
wv, weights = gausslegendre(300)
wv = (wv .+ 1) ./ 2 * (w_interval[2] - w_interval[1]) .+ w_interval[1]
c = (w_interval[2] - w_interval[1]) / 2

ks(t) = c * sum(S.(wv) .* cos.(2 * π * wv * t) .* weights);
nothing #hide
````

## Results
We see that we indeed recover the kernel from the spectral density,
with only a small error from the numerical integration.

````@example script
f = Figure(; size=(600, 400))
ax = Axis(f[1, 1]; xlabel="τ", ylabel="k(τ)")
lines!(ax, τv, k.(τv); label="kernel")
lines!(ax, τv, ks.(τv); label="spectral approx")
axislegend()
f
DisplayAs.PNG(f) # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

