using KernelSpectralDensities
import KernelSpectralDensities: OperatorDecomposition
using Test
using LinearAlgebra

@testset "MO decomposition via SpectralDensity" begin
    @testset "IndependentMOKernel" begin
        ker = IndependentMOKernel(SqExponentialKernel())

        S1 = SpectralDensity(ker, 1)
        @test S1.d isa OperatorDecomposition
        @test S1.d.B isa UniformScaling
        @test rand(S1) isa Real

        S2 = SpectralDensity(ker, 2)
        @test S2.d isa OperatorDecomposition
        @test S2.d.B isa UniformScaling

        w = rand(S2, 10)
        @test size(w) == (2, 10)
    end

    @testset "IntrinsicCoregionMOKernel" begin
        B = [1.0 0.2; 0.2 0.7]
        ker = IntrinsicCoregionMOKernel(SqExponentialKernel(), B)

        S1 = SpectralDensity(ker, 1)
        @test S1.d isa OperatorDecomposition
        @test S1.d.B isa LowerTriangular{Float64}
        @test size(S1.d.B) == size(B)
        @test S1.d.B * S1.d.B' ≈ B
        @test rand(S1) isa Real

        S2 = SpectralDensity(ker, 2)
        @test S2.d isa OperatorDecomposition
        @test S2.d.B isa LowerTriangular{Float64}
        @test size(S2.d.B) == size(B)

        w = rand(S2, 10)
        @test size(w) == (2, 10)
    end
end
