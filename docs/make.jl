using KernelSpectralDensities
using Documenter

DocMeta.setdocmeta!(KernelSpectralDensities, :DocTestSetup, :(using KernelSpectralDensities); recursive=true)

makedocs(;
    modules=[KernelSpectralDensities],
    authors="Steffen Ridderbusch <steffen@robots.ox.ac.uk
> and contributors",
    repo="https://github.com/Crown421/KernelSpectralDensities.jl/blob/{commit}{path}#{line}",
    sitename="KernelSpectralDensities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Crown421.github.io/KernelSpectralDensities.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Crown421/KernelSpectralDensities.jl",
    devbranch="main",
)
