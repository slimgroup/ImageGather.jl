
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://slimgroup.github.io/ImageGather.jl/dev) 

# ImageGather.jl

This package implements image gather functions for seismic inversion and QC. We currently only implemented surface gathers via the double RTM method (put ref) for QC on inverted velocity models.

# Example

A simple example of surface image gather for a layered model can be found in `examples/layers_cig.jl`. This examples produces the following image gathers:

![Single CDP offset gather](./docs/img/cig_cdp.png)
: Offset gather for a good and bad background velocity model at different position along X.

![eismic line of offset gathers](./docs/img/cig_line.png)
: Stack of offset gather along the X direction showing the difference in flatness and alignmement for a goood and bad background velocity model.


# Contributions

Contributions are welcome.

# Authors

This package is developed and maintained by Mathias Louboutin <mlouboutin3@gatech.edu>