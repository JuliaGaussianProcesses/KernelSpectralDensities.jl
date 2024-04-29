module KernelSpectralDensities

using Reexport
@reexport using KernelFunctions
using Distributions
using LinearAlgebra


import Base: rand
using Random

export SpectralDensity

# write tests to verify spectral density via fourier transforms
# also add SpectralKernel (which can then be learned?)

include("base.jl")
include("expkernels.jl")

end
