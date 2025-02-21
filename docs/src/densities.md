# Spectral Densities

A kernel ``k(x,y)`` is described as *stationary* or *shift-invariant*, if it can be written as ``k(τ) = k(x-y)``, which means that it only depends on the difference between ``x`` and ``y`` but not their absolute values. 

For a given stationary kernel ``k(τ)``, the *spectral density* is its Fourier transform
```math
S(\omega) = \int_{-\infty}^{\infty} k(τ) e^{-2 \pi \omega^T \tau} d\tau
```


!!! note
    The exact form of the Fourier transform may change slightly between fields (see [Details and Options](https://reference.wolfram.com/language/ref/FourierTransform.html)). 
    This package uses the "signal processing" form above, as done in [this presentation](https://gpss.cc/gpss21/slides/Heinonen2021.pdf) by Markus Heinonen. However, [Rahimi & Recht](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) used the "classical physics" form, for example. 
    All options are equivalent, if used consistently.

## Shared interface
```@docs
    KernelSpectralDensities.SpectralDensity
    Base.rand(::KernelSpectralDensities.AbstractSpectralDensity, ::Int64)
```

## Supported Kernel
- Squared Exponential
- Any Matern Kernel

## Supported Transformations
- with_lengthscale