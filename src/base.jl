
abstract type AbstractSpectralDensity end

(S::AbstractSpectralDensity)(w) = error("Not implemented")

"""
    rand(S::AbstractSpectralDensity, [n::Int])

Generate a sample [n samples] from the spectral density `S`. 
For a spectral density over ``ω \\in R``, computing `n`` samples results in a `n` dimensional vector. 
For a spectral density over ``ω \\in R^d, d>1``, computing `n` samples results in a `d,n` matrix. 

# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k);

julia> rand(S, 1);
```
"""
rand(S::AbstractSpectralDensity, n::Int...) = rand(Random.default_rng(), S, n...)

# rand(S::AbstractSpectralDensity) = rand(Random.default_rng(), S)

rand(::AbstractSpectralDensity) = throw(ArgumentError("KernelFunctions.kernel are input dimension agnostic. You must explicitly specify it."))

"""
    SpectralDensity{K<:Kernel}(k::Kernel, dim::Int)
Spectral density for the kernel K for `dim` dimensional frequency space. 

# Definition
Given a stationary kernel ``k(x, x')`` the spectral density is the Fourier transform of ``k(τ) = k(x-x')``. 
It can be seen as a probablity density function over the frequency domain, and can be evaluated at any frequency `w`. 

# Examples
```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k, 1);

julia> S(0.0)
2.5066282746310002

julia> S = SpectralDensity(k, 2);

julia> S(zeros(2))
6.283185307179585
```
"""
struct SpectralDensity{K<:KernelFunctions.Kernel} <: AbstractSpectralDensity
    kernel::K
    dim::Int
end

