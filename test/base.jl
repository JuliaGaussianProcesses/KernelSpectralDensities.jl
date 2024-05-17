using KernelSpectralDensities
using Test

@testset "Fallback" begin
    ker = ZeroKernel()

    S = SpectralDensity(ker, 1)

    @test_throws ErrorException S(1.0)
end

@testset "Dimension check" begin
    ker = ZeroKernel()

    @test_throws ArgumentError SpectralDensity(ker, 0)
end

@testset "Base rng" begin
    ker = SqExponentialKernel()

    @testset "1D" begin
        S = SpectralDensity(ker, 1)

        @test length(rand(S)) == 1
    end

    @testset "2D" begin
        S = SpectralDensity(ker, 2)

        @test length(rand(S)) == 2
    end
end
