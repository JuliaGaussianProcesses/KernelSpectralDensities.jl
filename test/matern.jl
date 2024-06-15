include("test_utils.jl")

if !isdefined(Base, :RUN_TESTS) || !RUN_TESTS
    using CairoMakie
    show_plot = true
else
    show_plot = false
end

@testset "Matern32" begin
    ker = Matern32Kernel()
    @testset "1D" begin
        @testset "Pure" begin
            w_interval = [-1.5, 1.5]
            t_interval = [0.0, 4.0]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end

        @testset "Scaled" begin
            ker = Matern32Kernel()
            ker = with_lengthscale(ker, 0.7)
            w_interval = [-2.0, 2.0]
            t_interval = [0.0, 3.5]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end
    end

    @testset "2D" begin
        @testset "Pure" begin
            ker = Matern32Kernel()
            w_interval = [-1.5, 1.5]
            x_interval = [-3.0, 3.0]

            test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end

        @testset "Scaled" begin
            ker = Matern32Kernel()
            ker = with_lengthscale(ker, 0.6)
            w_interval = [-2.0, 2.0]
            x_interval = [-3.5, 3.5]

            f = test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end
    end
end

@testset "Matern52" begin
    ker = Matern52Kernel()
    @testset "1D" begin
        @testset "Pure" begin
            ker = Matern52Kernel()
            w_interval = [-1.0, 1.0]
            t_interval = [0.0, 4.0]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end

        @testset "Scaled" begin
            ker = Matern52Kernel()
            ker = with_lengthscale(ker, 0.7)
            w_interval = [-1.5, 1.5]
            t_interval = [0.0, 3.5]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end
    end

    @testset "2D" begin
        @testset "Pure" begin
            ker = Matern52Kernel()
            w_interval = [-1.1, 1.1]
            x_interval = [-3.0, 3.0]

            test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end
    end
end

@testset "Matern12" begin
    ker = MaternKernel(; ν=0.5)
    @testset "1D" begin
        @testset "Pure" begin
            ker = MaternKernel(; ν=0.5)
            w_interval = [-3.0, 3.0]
            t_interval = [0.2, 6.0]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end

        @testset "Scaled" begin
            ker = MaternKernel(; ν=0.5)
            ker = with_lengthscale(ker, 0.7)
            w_interval = [-2.5, 2.5]
            t_interval = [0.0, 3.5]

            test_spectral_density(ker, w_interval, t_interval; show_plot)
        end
    end

    @testset "2D" begin
        @testset "Pure" begin
            ker = Matern52Kernel()
            w_interval = [-1.1, 1.1]
            x_interval = [-3.0, 3.0]

            test_2Dspectral_density(ker, w_interval, x_interval; show_plot)
        end
    end
end