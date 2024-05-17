using KernelSpectralDensities
using LinearAlgebra, StatsBase, AbstractGPs
using Test
using StableRNGs

ker = SqExponentialKernel()

# basic tests

function basic_gp_aprox_tests(rff, S, n_w)
    agps = ApproximateGPSample(rff(S, n_w))

    agps2 = ApproximateGPSample(S, n_w)

    @test length(agps.w) == length(agps2.w) == n_w

    if rff == ShiftedRFF
        @test typeof(agps.rff) == typeof(agps2.rff) == rff
    elseif rff == DoubleRFF
        @test typeof(agps.rff) != typeof(agps2.rff)
    end
end

@testset "basic" begin
    n_w = 100

    @testset "ShiftedRFF" begin
        rff = ShiftedRFF

        @testset "1D" begin
            d = 1
            S = SpectralDensity(ker, d)
            basic_gp_aprox_tests(rff, S, n_w)
        end

        @testset "2D" begin
            d = 2
            S = SpectralDensity(ker, d)
            basic_gp_aprox_tests(rff, S, n_w)
        end
    end

    @testset "DoubleRFF" begin
        rff = DoubleRFF

        @testset "1D" begin
            d = 1
            S = SpectralDensity(ker, d)
            basic_gp_aprox_tests(rff, S, n_w)
        end

        @testset "2D" begin
            d = 2
            S = SpectralDensity(ker, d)
            basic_gp_aprox_tests(rff, S, n_w)
        end
    end
end

# accuracy
function test_gp_approx_accuracy(rff, S, gp, x_eval, n_fs)
    rng = StableRNG(1234)
    function gp_approx_error(rng, rff, S, n_w, gp, x_eval, n_fs)
        f_samples = [ApproximateGPSample(rng, rff(rng, S, n_w)).(x_eval) for _ in 1:n_fs]

        m_sample = mean(f_samples)
        c_sample = cov(f_samples)
        # println(c_sample)

        m_exact = mean(gp, x_eval)
        c_exact = cov(gp, x_eval)
        # println(c_exact)

        mean_err = norm(m_sample .- m_exact)
        # cerr = (c_sample .- c_exact) ./ c_exact
        # cov_err = norm(cerr, Inf)
        cov_err = norm(c_sample .- c_exact, 2)

        return mean_err, cov_err
    end

    errs = [gp_approx_error(rng, rff, S, i, gp, x_eval, n_fs) for i in [2, 100, 100]]

    @test norm(getindex.(errs, 1), Inf) < 0.02

    @test all(diff(getindex.(errs, 2)) .< 0)
    # println(getindex.(errs, 2))
    @test getindex(errs, 2)[end] < 0.01
end

@testset "accuracy" begin
    gp = GP(ker)
    S = SpectralDensity(ker, 1)

    n_fs = 1e6

    x_eval = [0.5, 1.0, 2.0]
    @testset "ShiftedRFF" begin
        rff = ShiftedRFF
        test_gp_approx_accuracy(rff, S, gp, x_eval, n_fs)
    end

    @testset "DoubleRFF" begin
        rff = DoubleRFF
        test_gp_approx_accuracy(rff, S, gp, x_eval, n_fs)
    end
end
