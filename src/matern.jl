
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
    n = length(l)
    l = inv.(2 * π * l) .^ 2
    D = Distributions.MvTDist(2 * ν, zeros(n), diagm(l))
    return D
end
