export ShiftedRFF, DoubleRFF

"""
    AbstractRFF
Abstract type defining a random Fourier feature function. 
"""
abstract type AbstractRFF end

"""
    ShiftedRFF(S::SpectralDensity, l::Int)
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

julia> S = SpectralDensity(k);

julia> rff = ShiftedRFF(S, 2);

julia> rff(1.);
```
"""
struct ShiftedRFF <: AbstractRFF
    wv::Vector{Float64}
    b::Vector{Float64}

    function ShiftedRFF(S::SpectralDensity, l::Int)
        wv = [rand(S, 1) for _ in 1:l]
        b = rand(l)
        new(wv, b)
    end
end

function (rff::ShiftedRFF)(x)
    l = length(rff.wv)
    sqrt(2 / l) * cos.(2 * pi * ((rff.wv .* x) .+ rff.b))
end

"""
    DoubleRFF(S::SpectralDensity, l::Int)
Random Fourier feature function with cos and sin terms, projecting an input x into 2*l dimensionional feature space.

# Definition

This feature function is defined as
```math
    \\sqrt{1 / l} [\\cos(2 π w' x),  \\sin(2 π w' x)]
```
where `w` sampled from the spectral density S and `l` is the number of sampled frequencies. 
The output will be the result of `[cos(...w_1), cos(...w_2), ..., cos(...w_l), sin(...w_1), sin(...w_2), ..., sin(...w_l)]`.

# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k);

julia> rff = DoubleRFF(S, 2);

julia> rff(1.);

"""
struct DoubleRFF <: AbstractRFF
    # need to also deal with vector weights
    wv::Vector{Float64}

    function DualRFF(S::SpectralDensity, l::Int)
        wv = [rand(S, 1) for _ in 1:l]
        new(wv)
    end
end

function (rff::DoubleRFF)(x)
    l = length(rff.wv)
    # sqrt(1 / l) * (cos.(2 * pi * ((rff.wv .* x))) + sin.(2 * pi * ((rff.wv .* x))))
    c = cos.(2 * pi * ((rff.wv .* x)))
    s = sin.(2 * pi * ((rff.wv .* x)))
    sqrt(1 / l) * vcat(c, s)
end