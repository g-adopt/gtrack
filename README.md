# gtrack

**GPlates-based Tracking of Lithosphere and Kinematics**

A Python package for computing lithospheric structure through geological time using plate tectonic reconstructions.

## From plate reconstruction to model fields

Gtrack creates labelled source-point clouds through geological time. A
`LithosphereCloudSource` combines oceanic points from the seafloor-age tracker
with continental points reconstructed from their present-day positions. Each
point carries physical channels such as lithospheric thickness and material
age. G-ADOPT interpolates these channels onto its finite-element mesh and uses
them to construct indicator fields and geotherms.

Young oceanic lithosphere near spreading centres is thin, and its contribution
to a layer indicator can require a smooth lateral transition. To this end,
`OceanicLidWeight` maps oceanic thickness to a dimensionless `lateral_weight`
between zero and one. Continental source points retain a weight of one. In
G-ADOPT, `SourceLateralWeight` interpolates this channel and multiplies it by
the radial layer indicator.

Bounded regions use two separate channels. The `membership` channel describes
the lateral extent, while `masked_thickness` contains `membership * thickness`.
After interpolation, G-ADOPT divides the latter by nonzero membership to
recover physical thickness. This separation prevents the interpolation width
from changing a constant physical base depth across a region boundary.

## Installation

```bash
pip install gtrack
```

## Documentation

For full documentation, examples, and API reference, visit:

**[https://gtrack.gadopt.org](https://gtrack.gadopt.org)**

## License

MIT License
