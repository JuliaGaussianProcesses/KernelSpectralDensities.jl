using KernelSpectralDensities
using Test
using LinearAlgebra
using FastGaussQuadrature
using StatsBase
using StableRNGs

function test_spectral_density(ker::Kernel, w_interval, t_interval; show_plot::Bool=false)
    rng = StableRNG(123)
    S = SpectralDensity(ker, 1)
    k(t) = ker(0, t)

    # tbd on 2D
    τv = range(t_interval...; length=50)

    wv, weights = gausslegendre(400)
    wv = (wv .+ 1) ./ 2 * (w_interval[2] - w_interval[1]) .+ w_interval[1]
    c = (w_interval[2] - w_interval[1]) / 2

    ks(t) = c * sum(S.(wv) .* cos.(2 * π * wv * t) .* weights)

    @test norm(k.(τv) .- ks.(τv)) < 5e-3

    w_samples = rand(rng, S, Int(1e6))
    w_bins = range(w_interval...; length=100)
    h = fit(Histogram, w_samples, w_bins)
    h = normalize(h; mode=:pdf)

    midpoints = (h.edges[1][1:(end - 1)] .+ h.edges[1][2:end]) ./ 2
    @test norm(S.(midpoints) .- h.weights) < 0.1

    w_samples = rand(rng, S, Int(1e5))

    kss(t) = 1 / length(w_samples) * sum(cos.(2 * π * w_samples * t))

    @test norm(k.(τv) .- kss.(τv)) < 0.1

    if show_plot
        f = Figure()
        ax = Axis(f[1, 1])
        lines!(ax, τv, k.(τv); label="kernel")
        lines!(ax, τv, ks.(τv); label="spectral approx")
        axislegend()

        ax2 = Axis(f[1, 2])
        lines!(ax2, wv, S.(wv); label="spectral density")
        barplot!(
            ax2,
            midpoints,
            h.weights;
            strokewidth=0,
            strokedash=0,
            alpha=0.5,
            label="samples",
        )
        display(f)
    end
end

function test_2Dspectral_density(ker::Kernel, w_interval, x_interval; show_plot::Bool=false)
    rng = StableRNG(12345)
    S = SpectralDensity(ker, 2)
    k(x) = ker([0, 0], x)

    x = range(x_interval...; length=41)
    X = [[x_i, x_j] for x_i in x, x_j in x]
    # reference
    K = [k(X_i) for X_i in X]

    ## recover kernel from density
    # Gauss quadrature
    wv, weights = gausslegendre(250)
    wv = (wv .+ 1) ./ 2 * (w_interval[2] - w_interval[1]) .+ w_interval[1]
    weights = [w_i * w_j for w_i in weights, w_j in weights]
    Wv = [[w_i, w_j] for w_i in wv, w_j in wv]
    c = (w_interval[2] - w_interval[1]) / 2

    ks(t) = c^2 * sum(S.(Wv[:]) .* cos.(2 * π * dot.(Wv[:], [t])) .* weights[:])

    Ks = [ks(X_i) for X_i in X]
    @test norm(K .- Ks) < 0.015

    ## check sampling
    # w = range(w_interval..., length=80)
    # W = [[w_i, w_j] for w_i in w, w_j in w]

    w_samples = rand(rng, S, Int(1e7))
    # reshape to fit into histogram fit
    # w_tmp = reduce(hcat, w_samples[:])
    # w_tmp = (w_tmp[1, :], w_tmp[2, :])
    w_tmp = (w_samples[1, :], w_samples[2, :])

    w_bins = range(w_interval...; length=100)
    h = fit(Histogram, w_tmp, (w_bins, w_bins))
    h = normalize(h; mode=:pdf)
    midpoints1 = (h.edges[1][1:(end - 1)] .+ h.edges[1][2:end]) ./ 2
    midpoints2 = (h.edges[2][1:(end - 1)] .+ h.edges[2][2:end]) ./ 2

    midpoints = [[m1, m2] for m1 in midpoints1, m2 in midpoints2]
    @test norm(S.(midpoints) .- h.weights, Inf) < 0.5

    ## check kernel recovery from sammpling
    w_samples = rand(rng, S, Int(5e4))
    kss(x) = 1 / size(w_samples, 2) * sum(cos.(2 * π * dot.(eachcol(w_samples), [x])))

    Kss = [kss(X_i) for X_i in X]

    #ToDo: better would probably be to test if error decreases with more samples
    @test norm(K .- Kss) < 0.3

    if show_plot
        f = Figure(; size=(900, 1000))
        ax = Axis3(f[1, 1])
        contour3d!(
            ax,
            midpoints1,
            midpoints2,
            S.(midpoints);
            levels=11,
            color=:orangered3,
            linewidth=2.5,
            label="spectral density",
        )
        contour3d!(
            ax,
            midpoints1,
            midpoints2,
            h.weights;
            levels=11,
            color=:midnightblue,
            linewidth=2.5,
            linestyle=:dash,
            label="samples",
        )
        axislegend(ax)

        ax2 = Axis3(f[2, 1])
        contour3d!(ax2, x, x, K; levels=11, color=:orangered3, label="kernel")
        contour3d!(
            ax2,
            x,
            x,
            Ks;
            levels=11,
            color=:midnightblue,
            linewidth=2.5,
            linestyle=:dash,
            label="spectral approx",
        )
        contour3d!(
            ax2,
            x,
            x,
            Kss;
            levels=11,
            color=:darkgreen,
            linewidth=2.5,
            linestyle=:dashdot,
            label="spectral approx (sample)",
        )
        axislegend(ax2)

        display(f)
    end
end
