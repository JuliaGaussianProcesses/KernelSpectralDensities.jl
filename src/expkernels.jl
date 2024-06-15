
################################################### 
## Squared ExponentialKernel

# ToDo: Not sure about distances? Do all work?

function _spectral_distribution(ker::SqExponentialKernel, l::Real)
    return inv(2 * π * l) * Normal()
end

function _spectral_distribution(ker::SqExponentialKernel, l::AbstractVector)
    σv = abs2.(inv.(2 * π * l))
    return MvNormal(Diagonal(σv))
end
