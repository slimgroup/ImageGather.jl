using Documenter, ImageGather, DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "cig-refs.bib"), sorting = :nyt)

makedocs(bib, sitename="Image gather tools",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any["Home" => "index.md"])

deploydocs(repo="github.com/slimgroup/ImageGather.jl",
           devbranch="main")