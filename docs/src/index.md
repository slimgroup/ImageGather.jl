# ImageGather.jl documentation

ImageGather.jl provides computational QC tools for wave-equation based inversion. Namely we provides two main widely used workflows:

- Surface offset gathers (also called surface common image gather). Surface gather compute images of (RTMs) for different offset to verify the accuracy of the background velocity. The method we implement here is based on the double-rtm method [Giboli](@cite) that allows the computation of the gather with two RTMs only instead of one per offset (or offset bin).
- Subsurface offset gathers (also called subsurface common image gather) [sscig](@cite).

## Surface offset gathers

```@docs
surface_gather
double_rtm_cig
```

## Subsurface offset gathers

```@docs
judiExtendedJacobian
```

## Utility functions

```@autodocs
Modules = [ImageGather]
Pages = ["utils.jl"]
```

# References

```@bibliography
```


