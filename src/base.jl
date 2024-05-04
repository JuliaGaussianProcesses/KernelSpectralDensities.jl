
abstract type AbstractSpectralDensity end

(S::AbstractSpectralDensity)(w) = error("Not implemented")

"""
    rand(S::AbstractSpectralDensity, d::Int)

Generate a random d-dimensional sample from the spectral density `S`. 
Because many KernelFunctions.Kernel are input dimension agnostic, you must explicitly specify the input dimension `d`.

# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k);

julia> rand(S, 1)
```
"""
rand(S::AbstractSpectralDensity, d::Int) = rand(Random.default_rng(), S, d)

rand(::AbstractSpectralDensity) = throw(ArgumentError("KernelFunctions.kernel are input dimension agnostic. You must explicitly specify it."))

"""
    SpectralDensity{K<:Kernel}
Spectral density for the kernel K. 

# Definition
Given a stationary kernel ``k(x, x')`` the spectral density is the Fourier transform of ``k(τ) = k(x-x')``. 
It can be seen as a probablity density function over the frequency domain, and can be evaluated at any frequency `w`. 

# Examples
```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k);

julia> S(0.0)
2.5066282746310002
```
"""
struct SpectralDensity{K<:KernelFunctions.Kernel} <: AbstractSpectralDensity
    kernel::K
end

