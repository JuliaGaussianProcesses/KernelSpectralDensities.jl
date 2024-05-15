using KernelSpectralDensities
using Test
using LinearAlgebra
using FastGaussQuadrature
using StatsBase

@info "Packages Loaded"

include("test_utils.jl")

include("expkernels.jl")

include("features.jl")

include("approx_prior.jl")

