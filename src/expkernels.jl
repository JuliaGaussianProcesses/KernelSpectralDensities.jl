
################################################### 
## Squared ExponentialKernel

# ToDo: Not sure about distances? Do all work?
function _spectral_distribution(::SqExponentialKernel, l::Real)
    return INVPI * l * Normal()
end

function _spectral_distribution(::SqExponentialKernel, L::AbstractMatrix)
    # σv = abs2.(inv.(2 * π * l))
    σv = INVPI * L
    return MvNormal(σv' * σv)
end
