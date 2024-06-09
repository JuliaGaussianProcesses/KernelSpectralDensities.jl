
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
2.5066282746310002

julia> S = SpectralDensity(k, 2);

julia> S(zeros(2))
6.283185307179585
```
"""
struct SpectralDensity{K<:KernelFunctions.Kernel,D<:Distribution} <: AbstractSpectralDensity
    kernel::K
    # dim::Int
    d::D

    function SpectralDensity(kernel::KernelFunctions.Kernel, dim::Int)
        if dim < 1
            throw(ArgumentError("Dimension must be greater than 0"))
        end

        sk, l = _deconstruct_kernel(kernel)
        if dim == 1
            d = _spectral_distribution(sk, l)
        else
            d = _spectral_distribution(sk, l, dim)
        end
        return new{typeof(kernel),typeof(d)}(kernel, d)
    end
end

function _spectral_distribution(ker::KernelFunctions.Kernel, l)
    return throw(MethodError(_spectral_distribution, (ker,)))
end

function _spectral_distribution(ker::KernelFunctions.Kernel, l, d::Int=1)
    return throw(MethodError(_spectral_distribution, (ker, d)))
end

# ToDo: This could perhaps go into a separate file
# I could add `dim` here, and directly return l in the "right shape"?
# This would be either a vector or number (which would be great)
function _deconstruct_kernel(ker::SimpleKernel)
    return ker, 1.0
end

function _deconstruct_kernel(ker::TransformedKernel{<:SimpleKernel,<:ScaleTransform})
    return ker.kernel, 1 / ker.transform.s
end

function _deconstruct_kernel(ker::TransformedKernel)
    return throw(MethodError(_deconstruct_kernel, (ker,)))
end
