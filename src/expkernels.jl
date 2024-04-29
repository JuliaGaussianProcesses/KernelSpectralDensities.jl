

################################################### 
## Squared ExponentialKernel

# ToDo: Not sure about distances? Do all work?
(S::SpectralDensity{<:SqExponentialKernel})(w) = _sqexp(w, 1)

function (S::SpectralDensity{<:TransformedKernel{<:SqExponentialKernel,<:ScaleTransform}})(w)
    l = 1 / only(S.kernel.transform.s)
    _sqexp(w, l^2)
end

_sqexp(w::Real, l2::Real) = sqrt(2 * l2 * π) * exp(-2 * l2 * π^2 * w^2)
function _sqexp(w::AbstractVector{<:Real}, l2::Real)
    d = length(w)
    sqrt(2 * l2 * π)^d * exp(-2 * l2 * π^2 * dot(w, w))
end

function rand(rng::AbstractRNG, ::SpectralDensity{<:SqExponentialKernel}, d::Int)
    _sqexprand(rng, d, 1)
end

function rand(rng::AbstractRNG, S::SpectralDensity{<:TransformedKernel{<:SqExponentialKernel,<:ScaleTransform}}, d::Int)
    l = 1 / only(S.kernel.transform.s)
    _sqexprand(rng, d, l)
end

function _sqexprand(rng::AbstractRNG, d::Int, l::Real)
    σ = 1 / (2 * l * π)
    if d == 1
        return rand(rng, Normal(0, σ))
    elseif d > 1
        σv = ones(d) * abs2(σ)
        return rand(rng, MvNormal(Diagonal(σv)))
    else
        throw(ArgumentError("Number of input features must be greater than 0."))
    end
end


