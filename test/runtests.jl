using KernelSpectralDensities
using Test
using LinearAlgebra
using FastGaussQuadrature
using StatsBase
using StableRNGs

const RUN_TESTS = true

@info "Packages Loaded"

include("test_utils.jl")

@testset "Base" begin
    include("base.jl")
end

@testset "Decomposition" begin
    include("decomposition.jl")
end

@testset "Feature functions" begin
    include("features.jl")
end

@testset "MO Kernels" begin
    include("mokernels.jl")
end

if "heavy" in ARGS
    @testset "SpectralDensities" begin
        include("expkernels.jl")

        include("matern.jl")
    end

    @testset "Approximate prior" begin
        include("approx_prior.jl")
    end
end
