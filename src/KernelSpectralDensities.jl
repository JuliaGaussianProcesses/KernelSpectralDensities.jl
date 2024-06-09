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

include("base.jl")
include("expkernels.jl")
include("features.jl")
include("approx_prior.jl")

end
