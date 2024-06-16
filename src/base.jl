
abstract type AbstractSpectralDensity end

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
2.5066282746310007

julia> S = SpectralDensity(k, 2);

julia> S(zeros(2))
6.2831853071795845
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

        sk, l = _deconstruct_kernel(kernel, dim)
        d = _spectral_distribution(sk, l)

        return new{typeof(kernel),typeof(d)}(kernel, d)
    end
end

function (S::SpectralDensity)(w)
    return pdf(S.d, w)
end

function rand(rng::AbstractRNG, S::SpectralDensity, n::Int...)
    return rand(rng, S.d, n...)
end

# ToDo: This could perhaps go into a separate file
function _deconstruct_kernel(ker::KernelFunctions.SimpleKernel, dim::Int)
    if dim == 1
        l = 1.0
    else
        l = ones(dim)
    end
    return ker, l
end

function _deconstruct_kernel(
    ker::TransformedKernel{<:KernelFunctions.SimpleKernel,<:ScaleTransform}, dim::Int
)
    l = inv(only(ker.transform.s))
    if dim > 1
        l = ones(dim) * l
    end
    return ker.kernel, l
end

function _deconstruct_kernel(ker::TransformedKernel, dim::Int)
    return throw(MethodError(_deconstruct_kernel, (ker, dim)))
end

function _spectral_distribution(ker::KernelFunctions.Kernel, l)
    return throw(MethodError(_spectral_distribution, (ker,)))
end