using KernelSpectralDensities
using LinearAlgebra, StatsBase
using Test

function basic_rff_tests(RFF, ker, d)
    rng = StableRNG(1234)

    S = SpectralDensity(ker, d)

    n = 100
    rff = RFF(rng, S, n)

    eval_x = d == 1 ? rand(rng) : rand(rng, d)

    feats = rff(eval_x)

    @test length(feats) == n
end

function test_RFF_kernel_recovery(RFF, ker, d; plot=false)
    rng = StableRNG(12345)

    S = SpectralDensity(ker, d)

    x = zeros(d)

    n = 10
    r = range(0, 5; length=n)
    z = repeat([zeros(n)], d - 1)
    y = collect.(zip(r, z...))

    function kernel_approx_error(rff, ker, x, y)
        kfa(x, y) = dot(rff(x), rff(y))

        k1 = kfa.([x], y)
        k2 = ker.([x], y)

        return norm(k1 .- k2)
    end

    err = [kernel_approx_error(RFF(rng, S, Int(10^i)), ker, x, y) for i in (d + 1):(d + 3)]

    @test all(diff(err) .< 0)
    # println(diff(err))
    @test err[end] < 0.1
    # println(err[end])

    if plot
        f = Figure()
        ax = Axis(f[1, 1])

        lines!(ax, r, ker.([x], y); label="kernel")
        for i in 1:3
            rff = RFF(S, Int(10^i))
            lines!(ax, r, [dot(rff(x), rff(y_i)) for y_i in y]; label="rff approx $(10^i)")
        end
        axislegend()
        f
    end
end

@testset "ShiftedRFF" begin
    ker = SqExponentialKernel()
    RFF = ShiftedRFF

    @testset "basic" begin
        basic_rff_tests(RFF, ker, 1)
        basic_rff_tests(RFF, ker, 2)
    end

    @testset "kernel recovery" begin
        test_RFF_kernel_recovery(RFF, ker, 1)
        test_RFF_kernel_recovery(RFF, ker, 2)
    end
end

@testset "DoubleRFF" begin
    ker = SqExponentialKernel()
    RFF = DoubleRFF

    @testset "basic" begin
        basic_rff_tests(RFF, ker, 1)
        basic_rff_tests(RFF, ker, 2)
    end

    @testset "check error" begin
        S = SpectralDensity(ker, 1)
        @test_throws ArgumentError RFF(S, 1)
    end

    @testset "kernel recovery" begin
        test_RFF_kernel_recovery(RFF, ker, 1)
        test_RFF_kernel_recovery(RFF, ker, 2)
    end
end
