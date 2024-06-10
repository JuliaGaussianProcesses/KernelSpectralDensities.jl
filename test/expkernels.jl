include("test_utils.jl")

if !isdefined(Base, :RUN_TESTS) || !RUN_TESTS
    using CairoMakie
    show_plot = true
else
    show_plot = false
end

ker = Matern32Kernel()

@testset "SquaredExponential Kernel" begin
    ker = SqExponentialKernel()
    @testset "1D" begin
        @testset "Pure" begin
            w_interval = [-2.0, 2.0]
            t_interval = [0.0, 4.0]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end

        @testset "Scaled" begin
            ker = with_lengthscale(SqExponentialKernel(), 0.7)
            w_interval = [-2.0, 2.0]
            t_interval = [0.0, 4.0]

            f = test_spectral_density(ker, w_interval, t_interval; show_plot)
        end
    end

    @testset "2D" begin
        @testset "Pure" begin
            w_interval = [-2.0, 2.0]
            x_interval = [-2.0, 2.0]

            f = test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end

        @testset "Scaled" begin
            ker = with_lengthscale(SqExponentialKernel(), 0.7)
            w_interval = [-2.0, 2.0]
            x_interval = [-2.0, 2.0]

            f = test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end
    end
end
