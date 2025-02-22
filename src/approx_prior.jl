"""
    ApproximateGPSample(rff::AbstractRFF)
An approximate sample from the GP prior defined by the kernel that corresponds to the spectral density `S` that the RFF `rff` is based on.

# Definition

Using the a vector of `l` random fourier features `r(x)`, we can define the Bayesian linear model 
```math
    g_s(x) = w' r(x)
```
where `w_i ~ N(0, 1), i = 1,...,l`. 
Each draw of `w` results in a different function sample from the GP prior.


# Examples

```jldoctest
julia> k = SqExponentialKernel();

julia> S = SpectralDensity(k, 1);

julia> rff = DoubleRFF(S, 2);

julia> ap = ApproximateGPSample(rff);

julia> ap(1.);
```
"""
struct ApproximateGPSample{RFF<:AbstractRFF}
    w::Vector{Float64}
    rff::RFF

    function ApproximateGPSample(rng::AbstractRNG, rff::AbstractRFF)
        w = randn(rng, rff.l)
        return new{typeof(rff)}(w, rff)
    end
end

ApproximateGPSample(rff::AbstractRFF) = ApproximateGPSample(Random.default_rng(), rff)

function ApproximateGPSample(S::SpectralDensity, l::Int)
    rff = ShiftedRFF(S, l)
    return ApproximateGPSample(rff)
end

function (ap::ApproximateGPSample)(x)
    return dot(ap.rff(x), ap.w)
end
