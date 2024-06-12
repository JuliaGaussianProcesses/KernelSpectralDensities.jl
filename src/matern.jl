
################################################### 
## Matern Kernels

MaternKernels = Union{MaternKernel,Matern32Kernel,Matern52Kernel}

_matern_order(k::MaternKernel) = only(k.ν)
_matern_order(::Matern32Kernel) = 3 / 2
_matern_order(::Matern52Kernel) = 5 / 2

# rewrite everything as returning a distribution (kind of as originally planned)
# should be able to abstract/ generalize a lot of the special casing
function _spectral_distribution(kernel::MaternKernels, l::Real)
    ν = _matern_order(kernel)
    return inv(2 * π * l) * TDist(2 * ν)
end

function _spectral_distribution(kernel::MaternKernels, l::AbstractVector)
    ν = _matern_order(kernel)
    Dv = fill(TDist(2 * ν), length(l))
    return Product(inv.(2 * π * l) .* Dv)
end

# for "compatibility" with Distributions.jl, I think I only want a constructor for 
# the non-scaled version (i.e. MvTDist(), which sets mean to zero and Σ to eye)
# then set the others via + / * operations
struct MvTDist{T<:Real,Cov<:AbstractMatrix,Mean<:AbstractVector} <:
       Distributions.ContinuousMultivariateDistribution
    μ::Mean
    Σ::Cov
    ν::Int
end

import Base: length, eltype#, mean, var, cov

Base.length(d::MvTDist) = length(d.μ)

Base.eltype(::Type{<:MvTDist{T}}) where {T} = T

sampler(d::MvTDist) = d

Distributions._rand!(::AbstractRNG, d::MvTDist, x::AbstractArray) = d

# need to add SpecialFunctions.jl for loggamma
function Distributions._logpdf(d::MvTDist, x::AbstractArray)
    p = length(d)

    t1 = loggamma((d.ν + p) / 2) - loggamma(d.ν / 2)
    -p / 2 * (log(d.ν) + logπ)
    -1 / 2 * logdet(d.Σ)

    # log1p term not correct yet. Maybe use quadratic form from Kernelfunctions? 
    # also missing mu
    t2 = -(d.ν + p) / 2 * log1p(sum(x .* (d.Σ \ x)) / d.ν)

    return t1 + 1
end

mean(d::MvTDist) = d.μ

# suspect this is the diagonal
function var(d::MvTDist)
    return diag(cov(d))
end

function cov(d::MvTDist)
    if d.ν > 2
        return diag(d.Σ / (d.ν - 2))
    else
        ArgumentError("Variance is undefined for ν <= 2")
    end
end