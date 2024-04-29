
abstract type AbstractSpectralDensity end

(S::AbstractSpectralDensity)(w) = error("Not implemented")
rand(S::AbstractSpectralDensity, d::Int) = rand(Random.default_rng(), S, d)

rand(S::AbstractSpectralDensity) = throw(ArgumentError("KernelFunctions.kernel are input dimension agnostic. You must explicitly specify it."))

struct SpectralDensity{K<:KernelFunctions.Kernel} <: AbstractSpectralDensity
    kernel::K
end

