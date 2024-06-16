if (!@isdefined RUN_TESTS) || !RUN_TESTS
    using CairoMakie
    show_plot = true
    include("test_utils.jl")
else
    show_plot = false
end

@testset "SquaredExponential Kernel" begin
    ker = SqExponentialKernel()
    @testset "1D" begin
        @testset "Pure" begin
            # ker = SqExponentialKernel()
            w_interval = 2.0
            t_interval = [0.0, 4.0]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end

        @testset "Scaled" begin
            # ker = SqExponentialKernel()
            kert = with_lengthscale(ker, 0.7)
            w_interval = 2.0
            t_interval = [0.0, 4.0]

            f = test_spectral_density(kert, w_interval, t_interval; show_plot)
        end
    end

    @testset "2D" begin
        @testset "Pure" begin
            # ker = SqExponentialKernel()
            w_interval = [-2.0, 2.0]
            x_interval = [-2.0, 2.0]

            f = test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end

        @testset "Scaled" begin
            # ker = SqExponentialKernel()
            kert = with_lengthscale(ker, 0.7)
            w_interval = [-2.0, 2.0]
            x_interval = [-2.0, 2.0]

            f = test_2Dspectral_density(kert, w_interval, x_interval; show_plot)
        end
    end
end
