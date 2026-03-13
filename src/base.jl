struct OperatorDecomposition{B,D<:Distribution}
    B::B
    d::D
end

abstract type AbstractSpectralDensity end

"""
    rand(S::AbstractSpectralDensity, [n::Int])

Generate a sample [n samples] from the spectral density `S`. 
For a scalar spectral density over ``ω \\in R``, computing `n` samples results in a `n` dimensional vector. 
For a multi-dimensional spectral density over ``ω \\in R^d, d>1``, computing `n` samples results in a `d,n` matrix. 

# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k, 1);

julia> rand(S, 1);
```
"""
rand(S::AbstractSpectralDensity, n::Int...) = rand(Random.default_rng(), S, n...)

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
2.5066282746310007

julia> S = SpectralDensity(k, 2);

julia> S(zeros(2))
6.2831853071795845
```
"""
struct SpectralDensity{D,K<:KernelFunctions.Kernel} <: AbstractSpectralDensity
    d::D
    kernel::K
    # dim::Int

    function SpectralDensity(kernel::KernelFunctions.Kernel, dim::Int)
        if dim < 1
            throw(ArgumentError("Dimension must be greater than 0"))
        end

        d = _spectral_decomposition(kernel, dim)

        return new{typeof(d),typeof(kernel)}(d, kernel)
    end
end

function (S::SpectralDensity)(w)
    return pdf(S.d, w)
end

# ToDo: Implement this
function (S::SpectralDensity{<:OperatorDecomposition})(w)
    return false #pdf(S.d, w)
end

function rand(rng::AbstractRNG, S::SpectralDensity, n::Int...)
    return rand(rng, S.d, n...)
end

function rand(rng::AbstractRNG, S::SpectralDensity{<:OperatorDecomposition}, n::Int...)
    return rand(rng, S.d.d, n...)
end

INVPI = inv(2 * π)
