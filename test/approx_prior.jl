using KernelSpectralDensities
using LinearAlgebra, StatsBase, AbstractGPs
using Test


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

rff = DoubleRFF
# accuracy
function test_gp_approx_accuracy(rff, gp, x_eval, n_fs)
    function gp_approx_error(rff, gp, x_eval, n_fs)
        f_samples = [ApproximateGPSample(rff).(x_eval) for _ in 1:n_fs]

        m_sample = mean(f_samples)
        c_sample = cov(f_samples)

        m_exact = mean(gp, x_eval)
        c_exact = cov(gp, x_eval)

        mean_err = norm(m_sample .- m_exact)
        cov_err = norm(c_sample .- c_exact)

        return mean_err, cov_err
    end

    errs = [gp_approx_error(rff(S, 10^i), gp, x_eval, n_fs) for i in 1:3]

    @test all(getindex.(errs, 1) .< 0.01)

    @test all(diff(getindex.(errs, 2)) .< 0)
end

@testset "accuracy" begin
    gp = GP(ker)

    n_fs = 1e5

    x_eval = [0.5, 1.0, 2.0, 5.0]
    @testset "ShiftedRFF" begin
        rff = ShiftedRFF
        test_gp_approx_accuracy(rff, gp, x_eval, n_fs)
    end

    @testset "DoubleRFF" begin
        rff = DoubleRFF
        test_gp_approx_accuracy(rff, gp, x_eval, n_fs)
    end
end