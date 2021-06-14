using Documenter, ImageGather

makedocs(sitename="Image gather tools",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any["Home" => "index.md"])

deploydocs(repo="github.com/slimgroup/ImageGather.jl",
           devbranch="main")