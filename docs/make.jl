using KernelSpectralDensities
using Documenter

DocMeta.setdocmeta!(
    KernelSpectralDensities, :DocTestSetup, :(using KernelSpectralDensities); recursive=true
)

makedocs(;
    modules=[KernelSpectralDensities],
    authors="Steffen Ridderbusch <steffen@robots.ox.ac.uk> and contributors",
    repo="https://github.com/JuliaGaussianProcesses/KernelSpectralDensities.jl/blob/{commit}{path}#{line}",
    sitename="KernelSpectralDensities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliagaussianprocesses.github.io/KernelSpectralDensities.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Spectral Densities" => "densities.md",
        "Feature Functions" => "feature_functions.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelSpectralDensities.jl", 
    devbranch="main",
    push_preview = true
)
