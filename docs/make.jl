using Documenter, ImageGather, DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "cig-refs.bib"); style = :authoryear)

makedocs(;sitename="Image gather tools",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any["Home" => "index.md"],
         plugins=[bib])

deploydocs(repo="github.com/slimgroup/ImageGather.jl",
           devbranch="main")