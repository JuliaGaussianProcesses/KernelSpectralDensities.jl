module KernelSpectralDensities

using Reexport
@reexport using KernelFunctions
using Distributions
using LinearAlgebra


import Base: rand
using Random

export SpectralDensity
export ShiftedRFF, DoubleRFF
export ApproximateGPSample


# write tests to verify spectral density via fourier transforms
# also add SpectralKernel (which can then be learned?)

include("base.jl")
include("expkernels.jl")
include("features.jl")
include("approx_prior.jl")

end
