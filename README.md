# tdcalc
Compute 3D displacements and strains due to triangular dislocations using the artefact-free method of Nikkhoo and Walters, 2015.

[![DOI](https://zenodo.org/badge/269057031.svg)](https://zenodo.org/doi/10.5281/zenodo.12802953)

### Use of the code: 
The tdcalc functions generally requre four inputs: observation coordinates, triangle coordinates, slip amount per triangle, and the poisson's ratio. These inputs are defined as follows:

    obs: shape (n,3)
        Coordinates of calculation points in a local cartesian coordinate system, units of meters (East, North, Up). 
        Columns are X, Y and Z
    
    tri: shape (3,3)
        Coordinates of triangular dislocation vertices in the same coordinate system, units of meters.
        tri[0] contains [x,y,z] for the first vertex, etc.
    
    slip: shape(3,)
        Triangular dislocation slip vector components (Strike-slip, Dip-slip, Tensile-slip).
        Strike is defined as the horizontal co-planar vector in the triangle.
        If the element is horizontal, strike is in the Y-direction.
        
    nu: scalar
        Poisson's ratio.

For example, to generate half-space displacements from one triangle at two points:

    obs=np.array([[-3,-3,-3],[4,-3,0]])
    tri = np.array([[0, 0.1, -0.9],[1, -0.2, -1.2],[1, 1, -0.7]])
    slip=[1.3,1.4,1.5]
    nu=0.25
    displ=tdcalc.TDdispHS(obs,tri,slip,nu) 

The run_tdcalc.ipynb file contains a demonstration of how to use the scripts. If you wish to calculate displacements for many triangles, it is your responsibility to run the 

### Citation:
Please cite the Zenodo DOI if you use this code, for example:

Lindsey, E. O. (2024), tdcalc version 1.0, https://doi.org/10.5281/12802953.

Also cite:

Nikkhoo, M., & Walter, T. R. (2015). Triangular dislocation: an analytical, artefact-free solution. Geophysical Journal International, 201(2), 1119–1141. https://doi.org/10.1093/gji/ggv035
