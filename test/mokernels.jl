
function full_kernelmatrix_error(l, mdes, x1, x2, p)
    rff = MORFF(mdes, l; p)
    kapprox = rff(x1) * rff(x2)'

    mker = mdes.kernel
    mo1 = MOInput([x1], p)
    mo2 = MOInput([x2], p)
    kfull = kernelmatrix(mker, mo1, mo2)
    return norm(kfull - kapprox, 2)
end

function component_error(l, mdes, x1, x2, p; c2=1)
    x1mo = MOInput([x1], p)
    x2mo = MOInput([x2], p)
    morff = MORFF(mdes, l; p)
    mker = mdes.kernel
    return abs2(dot(morff(x1mo[1]), morff(x2mo[c2])) .- mker(x1mo[1], x2mo[c2]))
end

@testset "IndependentMOKernel" begin
    d = 2
    ker = SqExponentialKernel()
    mker = IndependentMOKernel(ker)
    mdes = SpectralDensity(mker, d)

    p = 3
    l = 2000

    x1 = rand(d)
    x2 = rand(d)

    kverr = [full_kernelmatrix_error(l, mdes, x1, x2, p) for l in [500, 5000]]
    @test all(diff(kverr) .< 0)

    cerr = [component_error(l, mdes, x1, x2, p) for l in [100, 5000]]
    @test all(diff(cerr) .< 0)

    c2err = component_error(l, mdes, x1, x2, p; c2=3)
    @test c2err ≈ 0
end
