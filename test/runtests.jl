using KernelSpectralDensities
using Test
using LinearAlgebra
using FastGaussQuadrature
using StatsBase

@info "Packages Loaded"

include("test_utils.jl")

@testset "SquaredExponential Kernel" begin
    ker = SqExponentialKernel()
    @testset "1D" begin
        @testset "Pure" begin
            w_interval = [-2.0, 2.0]
            t_interval = [0.0, 4.0]

            f = test_spectral_density(ker, w_interval, t_interval)
        end
        @info "1D Pure SqExp Done"

        @testset "Scaled" begin
            ker = with_lengthscale(SqExponentialKernel(), 0.7)
            w_interval = [-2.0, 2.0]
            t_interval = [0.0, 4.0]

            f = test_spectral_density(ker, w_interval, t_interval)
        end
        @info "1D Scaled SqExp Done"
    end

    @testset "2D" begin
        @testset "Pure" begin
            w_interval = [-2.0, 2.0]
            x_interval = [-2.0, 2.0]

            f = test_2Dspectral_density(ker, w_interval, x_interval)
        end
        @info "2D Pure SqExp Done"

        @testset "Scaled" begin
            ker = with_lengthscale(SqExponentialKernel(), 0.7)
            w_interval = [-2.0, 2.0]
            x_interval = [-2.0, 2.0]

            f = test_2Dspectral_density(ker, w_interval, x_interval)
        end
        @info "2D Scaled SqExp Done"
    end

end