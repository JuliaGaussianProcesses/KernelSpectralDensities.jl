
################################################### 
## Squared ExponentialKernel

# ToDo: Not sure about distances? Do all work?
(S::SpectralDensity{<:SqExponentialKernel})(w) = _sqexp(w, 1)

function (S::SpectralDensity{<:TransformedKernel{<:SqExponentialKernel,<:ScaleTransform}})(
    w
)
    l = 1 / only(S.kernel.transform.s)
    return _sqexp(w, l^2)
end

_sqexp(w::Real, l2::Real) = sqrt(2 * l2 * π) * exp(-2 * l2 * π^2 * w^2)
function _sqexp(w::AbstractVector{<:Real}, l2::Real)
    d = length(w)
    return sqrt(2 * l2 * π)^d * exp(-2 * l2 * π^2 * dot(w, w))
end

function rand(rng::AbstractRNG, S::SpectralDensity{<:SqExponentialKernel}, n::Int...)
    return _sqexprand(rng, S.dim, 1, n...)
end

function rand(
    rng::AbstractRNG,
    S::SpectralDensity{<:TransformedKernel{<:SqExponentialKernel,<:ScaleTransform}},
    n::Int...,
)
    l = 1 / only(S.kernel.transform.s)
    return _sqexprand(rng, S.dim, l, n...)
end

function _sqexprand(rng::AbstractRNG, d::Int, l::Real, n::Int...)
    σ = 1 / (2 * l * π)
    if d == 1
        return rand(rng, Normal(0, σ), n...)
    elseif d > 1
        σv = ones(d) * abs2(σ)
        return rand(rng, MvNormal(Diagonal(σv)), n...)
    end
end
