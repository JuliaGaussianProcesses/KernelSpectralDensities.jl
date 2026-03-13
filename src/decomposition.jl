function _base_l(dims::Int)
    if dims == 1
        return 1.0
    else
        return Diagonal(ones(dims))
    end
end

###################################
## Scalar kernels

function _spectral_decomposition(ker::KernelFunctions.SimpleKernel, dim::Int)
    return _spectral_distribution(ker, _base_l(dim))
end

SimpleTransforms = Union{IdentityTransform,ScaleTransform,ARDTransform}

function _spectral_decomposition(
    ker::TransformedKernel{<:KernelFunctions.SimpleKernel,<:SimpleTransforms}, dims::Int
)
    l = ker.transform(_base_l(dims))
    return _spectral_distribution(ker.kernel, l)
end

# maybe also LinearTransform? This one is more difficult....

###################################
## MO kernels

function _spectral_decomposition(ker::IndependentMOKernel, dim::Int)
    d = _spectral_decomposition(ker.kernel, dim)
    return OperatorDecomposition(I, d)
end

function _stackedB(B::UniformScaling, wv, p)
    if p == 0
        throw(ArgumentError("The output dimension p must be specified"))
    end
    return B[1:p, 1:p], (p, p)
end

function _spectral_decomposition(ker::IntrinsicCoregionMOKernel, dim::Int)
    d = _spectral_decomposition(ker.kernel, dim)
    B = cholesky(ker.B).L
    return OperatorDecomposition(B, d)
end

function _stackedB(B::Matrix, wv, p)
    # @assert size(B, 1) == p
    return B, size(B)
end
