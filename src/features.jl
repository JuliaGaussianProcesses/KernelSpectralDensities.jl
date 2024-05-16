"""
    AbstractRFF
Abstract type defining a random Fourier feature function. 
"""
abstract type AbstractRFF end

"""
    ShiftedRFF([rng::AbstractRNG], S::SpectralDensity, l::Int)
Random Fourier feature function with a random shift, projecting an input x into l dimensionional feature space. 
    
# Definition
    
This feature function is defined as 
```math
    \\sqrt{2 / l}  \\cos(2  π  ((w^T  x) + b))
```
where `w` sampled from the spectral density S, `b` is uniformly sampled from `[0, 2π]` and `l` is the number of sampled frequencies.

# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k, 1);

julia> rff = ShiftedRFF(S, 2);

julia> rff(1.);
```
"""
struct ShiftedRFF <: AbstractRFF
    wv::Union{Vector{<:Real},Matrix{<:Real}}
    b::Vector{Float64}
    l::Int

    function ShiftedRFF(rng::AbstractRNG, S::SpectralDensity, l::Int)
        w = rand(rng, S, l)
        b = rand(rng, l)
        new(w, b, l)
    end
end

ShiftedRFF(S::SpectralDensity, l::Int) = ShiftedRFF(Random.default_rng(), S, l)

_mul(w::Vector, x) = w .* x
_mul(w::Matrix, x) = dot.(eachcol(w), [x])

function (rff::ShiftedRFF)(x)
    sqrt(2 / rff.l) * cos.(2 * pi * (_mul(rff.wv, x) .+ rff.b))
end

"""
    DoubleRFF([rng::AbstractRNG], S::SpectralDensity, l::Int)
Random Fourier feature function with cos and sin terms, projecting an input x into l dimensionional feature space.

# Definition

This feature function is defined as
```math
    \\sqrt{1 / l} [\\cos(2 π w' x),  \\sin(2 π w' x)]
```
where `w` sampled from the spectral density S, with a total of `l/2` sampled frequencies.
The output will be the result of `[cos(...w_1), cos(...w_2), ..., cos(...w_l/2), sin(...w_1), sin(...w_2), ..., sin(...w_l/2)]`.

# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k, 1);

julia> rff = DoubleRFF(S, 2);

julia> rff(1.);
```
"""
struct DoubleRFF <: AbstractRFF
    wv::Union{Vector{<:Real},Matrix{<:Real}}
    l::Int

    function DoubleRFF(rng::AbstractRNG, S::SpectralDensity, l::Int)
        if l % 2 != 0
            throw(ArgumentError("l must be even"))
        end
        wv = rand(rng, S, div(l, 2))
        new(wv, l)
    end
end

DoubleRFF(S::SpectralDensity, l::Int) = DoubleRFF(Random.default_rng(), S, l)

function (rff::DoubleRFF)(x)
    c = cos.(2 * pi * (_mul(rff.wv, x)))
    s = sin.(2 * pi * (_mul(rff.wv, x)))
    sqrt(2 / rff.l) * vcat(c, s)
end