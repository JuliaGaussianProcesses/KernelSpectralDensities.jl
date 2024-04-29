function spectral_density(k::TransformedKernel{<:Kernel,<:ScaleTransform}, n_features)
    return spectral_density(k.kernel, n_features, k.transform.s[1])
end
function spectral_density(k::TransformedKernel{<:Kernel,<:ARDTransform}, n_features)
    return spectral_density(k.kernel, n_features, k.transform.v)
end
function spectral_density(mker::IndependentMOKernel, n_features)
    return spectral_density(mker.kernel, n_features)
end