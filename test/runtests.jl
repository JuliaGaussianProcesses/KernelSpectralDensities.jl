using KernelSpectralDensities
using Test
using LinearAlgebra
using FastGaussQuadrature
using StatsBase
using StableRNGs

@info "Packages Loaded"

include("test_utils.jl")

@testset "SpectralDensities" begin
    include("expkernels.jl")
end

@testset "Feature functions" begin
    include("features.jl")
end

@testset "Approximate prior" begin
    include("approx_prior.jl")
end

