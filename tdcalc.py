# tdcalc.py: Artefact-free triangular dislocations in python
#
# This package contains the following methods:
#
# TDdispFS(obs,tri,slip,nu): 
#    calculates displacements associated with a triangular dislocation
#    in an elastic full-space.
#
# TDdispHS(obs,tri,slip,nu): 
#    calculates displacements associated with a triangular dislocation
#    in an elastic full-space.
#
# TDstrainFS(obs,tri,slip,nu): 
#    calculates strains associated with a triangular dislocation 
#    in an elastic full-space.
#
# TDstrainHS(obs,tri,slip,nu): 
#    calculates strains associated with a triangular dislocation 
#    in an elastic full-space.
#
# strain2stress(strain,mu,lambda):
#    convert strains output by either TDstrainFS or TDstrainHS into stresses.
#    provided to match the outputs of the original MATLAB functions.
#
# TD: Triangular Dislocation
# EFCS: Earth-Fixed Coordinate System
# TDCS: Triangular Dislocation Coordinate System
# ADCS: Angular Dislocation Coordinate System
#    
# INPUTS
# obs: shape (n,3)
#    Coordinates of calculation points in EFCS (East, North, Up). 
#    Columns are X, Y and Z
#
#TODO: accept either orientation (n,3) or (3,n) for the obs points
#
# tri: shape (3,3)
#    Coordinates of TD vertices in EFCS.
#    tri[0] contains [x,y,z] for the first vertex, etc.
#
# slip: shape(3,)
#    TD slip vector components (Strike-slip, Dip-slip, Tensile-slip).
#    Strike is defined as the horizontal co-planar vector in the triangle.
#    If the element is horizontal, strike is in the Y-direction.
#
# nu: scalar
#    Poisson's ratio.
#
# OUTPUTS
# displacements [ue, un, uv]: shape (n,3)
#    Calculated displacement vector components in EFCS. ue, un and uv have
#    the same unit as slip in the inputs.
#
# strains [Exx,Eyy,Ezz,Exy,Exz,Eyz]: shape (n,6)
#    Calculated strain tensor components in EFCS. Dimensionless 
#    (Assumes the units of slip match the units of the observation coordinates).
#
# Modified from code presented in:
# Nikkhoo, M., Walter, T. R. (2015): Triangular dislocation: an analytical,
# artefact-free solution. - Geophysical Journal International, 201,
# 1117-1139. doi: 10.1093/gji/ggv035
#
# First Python version by Ben Thompson, 2018 (full-space displacements only)
# Modified by Eric Lindsey, 2020 (half-space displacements and strain)
#
#
# Original documentation license:
# Copyright (c) 2014 Mehdi Nikkhoo
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

########################################################################
################## General-purpose functions ###########################
########################################################################

def strain2stress(Exx,Eyy,Ezz,Exy,Exz,Eyz,mu,lam):
    ''' Convenience function, currently unused - original code output both Stress and Strain.'''
    # Calculate the stress tensor components in EFCS
    Sxx = 2*mu*Exx+lam*(Exx+Eyy+Ezz);
    Syy = 2*mu*Eyy+lam*(Exx+Eyy+Ezz);
    Szz = 2*mu*Ezz+lam*(Exx+Eyy+Ezz);
    Sxy = 2*mu*Exy;
    Sxz = 2*mu*Exz;
    Syz = 2*mu*Eyz;
    return Sxx,Syy,Szz,Sxy,Sxz,Syz

def normalize(v):
    return v / np.linalg.norm(v)

def CoordTrans(x,y,z,A):
    # convenience function to make this more similar to the matlab code
    trans = np.asarray(A).dot(np.asarray([x,y,z]))
    return trans[0], trans[1], trans[2]
    
def build_tri_coordinate_system(tri):
    # Calculate unit strike, dip and normal to TD vectors: For a horizontal TD 
    # as an exception, if the normal vector points upward, the strike and dip 
    # vectors point Northward and Westward, whereas if the normal vector points
    # downward, the strike and dip vectors point Southward and Westward, 
    # respectively.
    Vnorm = normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    eY = np.array([0, 1, 0])
    eZ = np.array([0, 0, 1])
    Vstrike = np.cross(eZ, Vnorm)
    if np.linalg.norm(Vstrike) == 0:
        Vstrike = eY * Vnorm[2]
        # TODO: check this correction from TDdispHS:
        #% For horizontal elements in case of half-space calculation!!!
        #% Correct the strike vector of image dislocation only
        if tri[0][2]>0:
            Vstrike = -Vstrike;
    Vstrike = normalize(Vstrike)
    Vdip = np.cross(Vnorm, Vstrike)
    return np.array([Vnorm, Vstrike, Vdip])

def trimodefinder(obs, tri):
    # trimodefinder calculates the normalized barycentric coordinates of
    # the points with respect to the TD vertices and specifies the appropriate
    # artefact-free configuration of the angular dislocations for the
    # calculations. The input matrices x, y and z share the same size and
    # correspond to the y, z and x coordinates in the TDCS, respectively. p1,
    # p2 and p3 are two-component matrices representing the y and z coordinates
    # of the TD vertices in the TDCS, respectively.
    # The components of the output (trimode) corresponding to each calculation
    # points, are 1 for the first configuration, -1 for the second
    # configuration and 0 for the calculation point that lie on the TD sides.
    
    a = ((tri[1,1]-tri[2,1])*(obs[0]-tri[2,0])+(tri[2,0]-tri[1,0])*(obs[1]-tri[2,1]))/ \
        ((tri[1,1]-tri[2,1])*(tri[0,0]-tri[2,0])+(tri[2,0]-tri[1,0])*(tri[0,1]-tri[2,1]))
    b = ((tri[2,1]-tri[0,1])*(obs[0]-tri[2,0])+(tri[0,0]-tri[2,0])*(obs[1]-tri[2,1]))/ \
        ((tri[1,1]-tri[2,1])*(tri[0,0]-tri[2,0])+(tri[2,0]-tri[1,0])*(tri[0,1]-tri[2,1]))
    c = 1-a-b
    
    trimode = np.ones(len(obs[0]),dtype=int)
    trimode[np.logical_and(np.logical_and(a<=0 , b>c) , c>a)] = -1
    trimode[np.logical_and(np.logical_and(b<=0 , c>a) , a>b)] = -1
    trimode[np.logical_and(np.logical_and(c<=0 , a>b) , b>c)] = -1
    trimode[np.logical_and(np.logical_and(a==0 , b>=0) , c>=0)] = 0
    trimode[np.logical_and(np.logical_and(a>=0 , b==0) , c>=0)] = 0
    trimode[np.logical_and(np.logical_and(a>=0 , b>=0) , c==0)] = 0
    trimode[np.logical_and(trimode==0 , obs[2]!=0)] = 1
    
    # flatnonzero is similar to np.where()[0] 
    # but this directly returns an array instead of tuple
    Ipos = np.flatnonzero(trimode==1) 
    Ineg = np.flatnonzero(trimode==-1)
    Inan = np.flatnonzero(trimode==0)
    
    return Ipos,Ineg,Inan

def TDtransform_pts_slip(obs,slip_b,TriVertex,SideVec):
    # Transform calculation points and slip vector components from TDCS into ADCS
    A = np.array([[SideVec[2], -SideVec[1]], [SideVec[1], SideVec[2]]])
    
    # Transform coordinates of the calculation points from TDCS into ADCS
    r1 = A.dot([obs[1]-TriVertex[1], obs[2]-TriVertex[2]])
    y1 = r1[0]
    z1 = r1[1]

    # Transform the in-plane slip vector components from TDCS into ADCS
    r2 = A.dot([slip_b[1], slip_b[2]])
    by1 = r2[0]
    bz1 = r2[1]
    
    return A,y1,z1,by1,bz1

def setupTDCS(obs,tri):
    # Transform coordinates from EFCS into TDCS
        
    # this contains the lengths, strikes, and dips for the triangle sides
    transform = build_tri_coordinate_system(tri)

    # note, transformed_obs will now have a shape of (3,n) instead of (n,3) but this is OK
    transformed_obs = transform.dot(np.transpose(obs - tri[1]))

    transformed_tri = np.zeros((3,3))
    transformed_tri[0,:] = transform.dot(tri[0] - tri[1])
    transformed_tri[2,:] = transform.dot(tri[2] - tri[1])
    
    np.testing.assert_almost_equal(transformed_tri[1], [0,0,0])
    np.testing.assert_almost_equal(transformed_tri[0][0], 0)
    np.testing.assert_almost_equal(transformed_tri[2][0], 0)

    # Calculate the unit vectors along TD sides in TDCS
    e12 = normalize(transformed_tri[1] - transformed_tri[0])
    e13 = normalize(transformed_tri[2] - transformed_tri[0])
    e23 = normalize(transformed_tri[2] - transformed_tri[1])

    # Calculate the TD angles
    A = np.arccos(e12.T.dot(e13))
    B = np.arccos(-e12.T.dot(e23))
    C = np.arccos(e23.T.dot(e13))

    return transform,transformed_obs,transformed_tri,e12,e13,e23,A,B,C

def TensTrans(Txx1,Tyy1,Tzz1,Txy1,Txz1,Tyz1,transform):
    # TensTrans Transforms the coordinates of tensors,from x1y1z1 coordinate
    # system to x2y2z2 coordinate system. "A" is the transformation matrix, 
    # whose columns e1,e2 and e3 are the unit base vectors of the x1y1z1. The 
    # coordinates of e1,e2 and e3 in A must be given in x2y2z2. The transpose 
    # of A (i.e., A.T) does the transformation from x2y2z2 into x1y1z1.
 
    # we access the elements of A directly. Note we transpose the ordering to match Matlab
    A=transform.T.ravel()
    
    Txx2 = A[0]**2*Txx1+2*A[0]*A[3]*Txy1+2*A[0]*A[6]*Txz1+2*A[3]*A[6]*Tyz1+A[3]**2*Tyy1+A[6]**2*Tzz1
    Tyy2 = A[1]**2*Txx1+2*A[1]*A[4]*Txy1+2*A[1]*A[7]*Txz1+2*A[4]*A[7]*Tyz1+A[4]**2*Tyy1+A[7]**2*Tzz1
    Tzz2 = A[2]**2*Txx1+2*A[2]*A[5]*Txy1+2*A[2]*A[8]*Txz1+2*A[5]*A[8]*Tyz1+A[5]**2*Tyy1+A[8]**2*Tzz1
    Txy2 = A[0]*A[1]*Txx1+(A[0]*A[4]+A[1]*A[3])*Txy1+(A[0]*A[7]+A[1]*A[6])*Txz1+\
           (A[7]*A[3]+A[6]*A[4])*Tyz1+A[4]*A[3]*Tyy1+A[6]*A[7]*Tzz1
    Txz2 = A[0]*A[2]*Txx1+(A[0]*A[5]+A[2]*A[3])*Txy1+(A[0]*A[8]+A[2]*A[6])*Txz1+\
           (A[8]*A[3]+A[6]*A[5])*Tyz1+A[5]*A[3]*Tyy1+A[6]*A[8]*Tzz1
    Tyz2 = A[1]*A[2]*Txx1+(A[2]*A[4]+A[1]*A[5])*Txy1+(A[2]*A[7]+A[1]*A[8])*Txz1+\
           (A[7]*A[5]+A[8]*A[4])*Tyz1+A[4]*A[5]*Tyy1+A[7]*A[8]*Tzz1
    return Txx2,Tyy2,Tzz2,Txy2,Txz2,Tyz2

########################################################################
################## Full-space displacement functions ###################
########################################################################

def TDdispFS(obs, tri, slip, nu):
    # calculates displacements associated with a triangular dislocation in an
    # elastic full-space.
    
    # require ndmin=2 in case of only 1 obs point being passed
    if np.ndim(obs)<2:
        obs=np.array(obs,ndmin=2)
    
    # define slip vector
    slip_b = np.array([slip[2], slip[0], slip[1]])
    
    # convert coordinates from EFCS to TDCS
    transform,transformed_obs,transformed_tri,e12,e13,e23,A,B,C = setupTDCS(obs,tri)
    
    # select appropriate angular dislocations for artefact-free solution
    Ipos,Ineg,Inan = trimodefinder(np.array([transformed_obs[1],\
                                             transformed_obs[2],transformed_obs[0]]),\
                                             transformed_tri[:,1:])
    
    # initialize output array - shape is (3,n) but note it will be transposed before output
    out = np.empty((3,len(transformed_obs[0])),dtype='float')
    
    if len(Ipos)>0:
        # Calculate first angular dislocation contribution
        u1Tp,v1Tp,w1Tp = TDSetupD(transformed_obs[:,Ipos],A,slip_b,nu,transformed_tri[0], -e13)
        # Calculate second angular dislocation contribution
        u2Tp,v2Tp,w2Tp = TDSetupD(transformed_obs[:,Ipos],B,slip_b,nu,transformed_tri[1], e12)
        # Calculate third angular dislocation contribution
        u3Tp,v3Tp,w3Tp = TDSetupD(transformed_obs[:,Ipos],C,slip_b,nu,transformed_tri[2], e23)
        out[:,Ipos] = np.array([
            u1Tp+u2Tp+u3Tp,
            v1Tp+v2Tp+v3Tp,
            w1Tp+w2Tp+w3Tp
        ])
    if len(Ineg)>0:
        # Calculate first angular dislocation contribution
        u1Tn,v1Tn,w1Tn = TDSetupD(transformed_obs[:,Ineg],A,slip_b,nu,transformed_tri[0],e13)
        # Calculate second angular dislocation contribution
        u2Tn,v2Tn,w2Tn = TDSetupD(transformed_obs[:,Ineg],B,slip_b,nu,transformed_tri[1],-e12)
        # Calculate third angular dislocation contribution
        u3Tn,v3Tn,w3Tn = TDSetupD(transformed_obs[:,Ineg],C,slip_b,nu,transformed_tri[2],-e23)
        out[:,Ineg] = np.array([
            u1Tn+u2Tn+u3Tn,
            v1Tn+v2Tn+v3Tn,
            w1Tn+w2Tn+w3Tn
        ])
    if len(Inan)>0:
        out[:,Inan] = np.nan
        
    a = np.array([
        -transformed_obs[0],
        transformed_tri[0][1] - transformed_obs[1],
        transformed_tri[0][2] - transformed_obs[2]
    ])
    b = -transformed_obs
    c = np.array([
        -transformed_obs[0],
        transformed_tri[2][1] - transformed_obs[1],
        transformed_tri[2][2] - transformed_obs[2]
    ])
    na = np.sqrt(np.sum(a**2,axis=0))
    nb = np.sqrt(np.sum(b**2,axis=0))
    nc = np.sqrt(np.sum(c**2,axis=0))
    
    FiN = (a[0]*(b[1]*c[2]-b[2]*c[1])- \
           a[1]*(b[0]*c[2]-b[2]*c[0])+ \
           a[2]*(b[0]*c[1]-b[1]*c[0]))
    FiD = na*nb*nc + np.sum(a*b,axis=0)*nc + np.sum(a*c,axis=0)*nb + np.sum(b*c,axis=0)*na
    Fi = -2*np.arctan2(FiN,FiD)/4/np.pi
    
    # Calculate the complete displacement vector components in TDCS
    out += np.outer(slip_b, Fi)
    
    # Transform the complete displacement vector components from TDCS into EFCS
    # also has the effect of rotating the shape of out from (3,n) to (n,3)
    # so that it matches the input obs array
    return out.T.dot(transform)

def TDSetupD(obs, alpha, slip_b, nu, TriVertex, SideVec):
    # TDSetupD transforms coordinates of the calculation points as well as
    # slip vector components from ADCS into TDCS. It then calculates the
    # displacements in ADCS and transforms them into TDCS.

    # Transform calculation points and slip vector components from TDCS into ADCS
    A,y1,z1,by1,bz1 = TDtransform_pts_slip(obs,slip_b,TriVertex,SideVec)
    
    # Calculate displacements associated with an angular dislocation in ADCS
    [u,v0,w0] = AngDisDisp(obs[0],y1,z1,-np.pi+alpha,slip_b[0],by1,bz1,nu)

    # Transform displacements from ADCS into TDCS
    r3 = A.T.dot([v0,w0])
    v = r3[0]
    w = r3[1]
    return u, v, w

def AngDisDisp(x, y, z, alpha, bx, by, bz, nu):
    # AngDisDisp calculates the "incomplete" displacements (without the
    # Burgers' function contribution) associated with an angular dislocation in
    # an elastic full-space.
    
    cosA = np.cos(alpha)
    sinA = np.sin(alpha)
    eta = y*cosA-z*sinA
    zeta = y*sinA+z*cosA
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    # Avoid complex results for the logarithmic terms
    Izeta = np.flatnonzero(zeta>r)
    Iz = np.flatnonzero(z>r)
    zeta[Izeta] = r[Izeta]
    z[Iz] = r[Iz]

    ux = bx/8/np.pi/(1-nu)*(x*y/r/(r-z)-x*eta/r/(r-zeta))
    vx = bx/8/np.pi/(1-nu)*(eta*sinA/(r-zeta)-y*eta/r/(r-zeta)+\
        y**2/r/(r-z)+(1-2*nu)*(cosA*np.log(r-zeta)-np.log(r-z)))
    wx = bx/8/np.pi/(1-nu)*(eta*cosA/(r-zeta)-y/r-eta*z/r/(r-zeta)-\
        (1-2*nu)*sinA*np.log(r-zeta))

    uy = by/8/np.pi/(1-nu)*(x**2*cosA/r/(r-zeta)-x**2/r/(r-z)-\
        (1-2*nu)*(cosA*np.log(r-zeta)-np.log(r-z)))
    vy = by*x/8/np.pi/(1-nu)*(y*cosA/r/(r-zeta)-sinA*cosA/(r-zeta)-y/r/(r-z))
    wy = by*x/8/np.pi/(1-nu)*(z*cosA/r/(r-zeta)-cosA**2/(r-zeta)+1/r)

    uz = bz*sinA/8/np.pi/(1-nu)*((1-2*nu)*np.log(r-zeta)-x**2/r/(r-zeta))
    vz = bz*x*sinA/8/np.pi/(1-nu)*(sinA/(r-zeta)-y/r/(r-zeta))
    wz = bz*x*sinA/8/np.pi/(1-nu)*(cosA/(r-zeta)-z/r/(r-zeta))

    return ux+uy+uz, vx+vy+vz, wx+wy+wz

########################################################################
################## Half-space displacement functions ###################
########################################################################
                                
def TDdispHS(obs,tri,slip,nu):
    # Calculates displacements associated with a triangular dislocation in an 
    # elastic half-space.
    
    # require ndmin=2 in case of only 1 obs point being passed
    if np.ndim(obs)<2:
        obs=np.array(obs,ndmin=2)

    assert all(obs[:,2]<=0), 'Half-space solution: observation Z coordinates must be zero or negative!'
    assert all(tri[:,2]<=0), 'Half-space solution: triangle Z coordinates must be zero or negative!'

    # Calculate main dislocation contribution to displacements
    uMS = TDdispFS(obs, tri, slip, nu)

    # Calculate harmonic function contribution to displacements
    uFSC = TDdisp_HarFunc(obs, tri, slip, nu)

    # Calculate image dislocation contribution to displacements
    tri_img = np.copy(tri) # do not modify tri!
    tri_img[:,2] = -tri_img[:,2]
    uIS = TDdispFS(obs, tri_img, slip, nu)
    if all(tri[:,2]==0):
        uIS[:,2] = -uIS[:,2]
    
    # Calculate the complete displacement vector components in EFCS
    u = uMS+uIS+uFSC
    if all(tri[:,2]==0):
        u = -u
    
    return u
                               
def TDdisp_HarFunc(obs,tri,slip,nu):
    # TDdisp_HarFunc calculates the harmonic function contribution to the
    # displacements associated with a triangular dislocation in a half-space.
    # The function cancels the surface normal tractions induced by the main and
    # image dislocations.

    bx = slip[2] # Tensile-slip
    by = slip[0] # Strike-slip
    bz = slip[1] # Dip-slip

    # Transform slip vector components from TDCS into EFCS
    A = build_tri_coordinate_system(tri)

    bX,bY,bZ = CoordTrans(bx,by,bz,A.T)
    
    # Calculate contribution of angular dislocation pair on each TD side 
    u1,v1,w1 = AngSetupFSC(obs,bX,bY,bZ,tri[0],tri[1],nu) # Side P1P2
    u2,v2,w2 = AngSetupFSC(obs,bX,bY,bZ,tri[1],tri[2],nu) # Side P2P3
    u3,v3,w3 = AngSetupFSC(obs,bX,bY,bZ,tri[2],tri[0],nu) # Side P3P1
    
    # Calculate total harmonic function contribution to displacements
    return np.array([u1+u2+u3, v1+v2+v3, w1+w2+w3]).T

def AngSetupFSC(obs,bX,bY,bZ,PA,PB,nu):
    # AngSetupFSC calculates the Free Surface Correction to displacements 
    # associated with angular dislocation pair on each TD side.
    
    npts=len(obs[:,0])
    
    # Calculate TD side vector and the angle of the angular dislocation pair
    SideVec = PB-PA
    eZ = np.array([0, 0, 1])
    beta = np.arccos(-SideVec.dot(eZ)/np.linalg.norm(SideVec))
    
    if (np.abs(beta) < np.finfo(float).eps or np.abs(np.pi-beta) < np.finfo(float).eps):
        ue = np.zeros(npts)
        un = np.zeros(npts)
        uv = np.zeros(npts)
    else:
        ey1 = normalize(np.array([SideVec[0],SideVec[1],0]))
        ey3 = -eZ
        ey2 = np.cross(ey3,ey1)
        A = np.array([ey1,ey2,ey3]) # Transformation matrix

        # Transform coordinates from EFCS to the first ADCS
        y1A,y2A,y3A = CoordTrans(obs[:,0]-PA[0],obs[:,1]-PA[1],obs[:,2]-PA[2],A)
        # Transform coordinates from EFCS to the second ADCS
        y1AB,y2AB,y3AB = CoordTrans(SideVec[0],SideVec[1],SideVec[2],A)
        y1B = y1A-y1AB
        y2B = y2A-y2AB
        y3B = y3A-y3AB

        # Transform slip vector components from EFCS to ADCS
        b1,b2,b3 = CoordTrans(bX,bY,bZ,A)

        # Determine the best arteact-free configuration for the calculation
        # points near the free furface
        Ipos = (beta*y1A >= 0)
        Ineg = np.logical_not(Ipos)
        
        v1A=np.empty(npts)
        v1B=np.empty(npts)
        v2A=np.empty(npts)
        v2B=np.empty(npts)
        v3A=np.empty(npts)
        v3B=np.empty(npts)
        
        # Configuration I
        v1A[Ipos],v2A[Ipos],v3A[Ipos] = AngDisDispFSC(y1A[Ipos],y2A[Ipos],y3A[Ipos],
                                                      -np.pi+beta,b1,b2,b3,nu,-PA[2])
        v1B[Ipos],v2B[Ipos],v3B[Ipos] = AngDisDispFSC(y1B[Ipos],y2B[Ipos],y3B[Ipos],
                                                      -np.pi+beta,b1,b2,b3,nu,-PB[2])

        # Configuration II
        v1A[Ineg],v2A[Ineg],v3A[Ineg] = AngDisDispFSC(y1A[Ineg],y2A[Ineg],y3A[Ineg],
                                                      beta,b1,b2,b3,nu,-PA[2])
        v1B[Ineg],v2B[Ineg],v3B[Ineg] = AngDisDispFSC(y1B[Ineg],y2B[Ineg],y3B[Ineg],
                                                      beta,b1,b2,b3,nu,-PB[2])

        # Calculate total Free Surface Correction to displacements in ADCS
        v1 = v1B-v1A
        v2 = v2B-v2A
        v3 = v3B-v3A

        # Transform total Free Surface Correction to displacements from ADCS to EFCS
        ue,un,uv = CoordTrans(v1,v2,v3,A.T)
        
    return ue,un,uv
                                
def AngDisDispFSC(y1,y2,y3,beta,b1,b2,b3,nu,a):
    # AngDisDispFSC calculates the harmonic function contribution to the 
    # displacements associated with an angular dislocation in an elastic 
    # half-space.

    sinB = np.sin(beta)
    cosB = np.cos(beta)
    cotB = 1.0/np.tan(beta)
    y3b = y3+2*a
    z1b = y1*cosB+y3b*sinB
    z3b = -y1*sinB+y3b*cosB
    r2b = y1**2+y2**2+y3b**2
    rb = np.sqrt(r2b)

    Fib = 2*np.arctan(-y2/(-(rb+y3b)/np.tan(beta/2)+y1)) # The Burgers' function

    v1cb1 = b1/4/np.pi/(1-nu)*(-2*(1-nu)*(1-2*nu)*Fib*cotB**2+(1-2*nu)*y2/\
        (rb+y3b)*((1-2*nu-a/rb)*cotB-y1/(rb+y3b)*(nu+a/rb))+(1-2*nu)*\
        y2*cosB*cotB/(rb+z3b)*(cosB+a/rb)+a*y2*(y3b-a)*cotB/rb**3+y2*\
        (y3b-a)/(rb*(rb+y3b))*(-(1-2*nu)*cotB+y1/(rb+y3b)*(2*nu+a/rb)+\
        a*y1/rb**2)+y2*(y3b-a)/(rb*(rb+z3b))*(cosB/(rb+z3b)*((rb*\
        cosB+y3b)*((1-2*nu)*cosB-a/rb)*cotB+2*(1-nu)*(rb*sinB-y1)*cosB)-\
        a*y3b*cosB*cotB/rb**2))

    v2cb1 = b1/4/np.pi/(1-nu)*((1-2*nu)*((2*(1-nu)*cotB**2-nu)*np.log(rb+y3b)-(2*\
        (1-nu)*cotB**2+1-2*nu)*cosB*np.log(rb+z3b))-(1-2*nu)/(rb+y3b)*(y1*\
        cotB*(1-2*nu-a/rb)+nu*y3b-a+y2**2/(rb+y3b)*(nu+a/rb))-(1-2*\
        nu)*z1b*cotB/(rb+z3b)*(cosB+a/rb)-a*y1*(y3b-a)*cotB/rb**3+\
        (y3b-a)/(rb+y3b)*(-2*nu+1/rb*((1-2*nu)*y1*cotB-a)+y2**2/(rb*\
        (rb+y3b))*(2*nu+a/rb)+a*y2**2/rb**3)+(y3b-a)/(rb+z3b)*(cosB**2-\
        1/rb*((1-2*nu)*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**3-1/(rb*\
        (rb+z3b))*(y2**2*cosB**2-a*z1b*cotB/rb*(rb*cosB+y3b))))

    v3cb1 = b1/4/np.pi/(1-nu)*(2*(1-nu)*(((1-2*nu)*Fib*cotB)+(y2/(rb+y3b)*(2*\
        nu+a/rb))-(y2*cosB/(rb+z3b)*(cosB+a/rb)))+y2*(y3b-a)/rb*(2*\
        nu/(rb+y3b)+a/rb**2)+y2*(y3b-a)*cosB/(rb*(rb+z3b))*(1-2*nu-\
        (rb*cosB+y3b)/(rb+z3b)*(cosB+a/rb)-a*y3b/rb**2))

    v1cb2 = b2/4/np.pi/(1-nu)*((1-2*nu)*((2*(1-nu)*cotB**2+nu)*np.log(rb+y3b)-(2*\
        (1-nu)*cotB**2+1)*cosB*np.log(rb+z3b))+(1-2*nu)/(rb+y3b)*(-(1-2*nu)*\
        y1*cotB+nu*y3b-a+a*y1*cotB/rb+y1**2/(rb+y3b)*(nu+a/rb))-(1-2*\
        nu)*cotB/(rb+z3b)*(z1b*cosB-a*(rb*sinB-y1)/(rb*cosB))-a*y1*\
        (y3b-a)*cotB/rb**3+(y3b-a)/(rb+y3b)*(2*nu+1/rb*((1-2*nu)*y1*\
        cotB+a)-y1**2/(rb*(rb+y3b))*(2*nu+a/rb)-a*y1**2/rb**3)+(y3b-a)*\
        cotB/(rb+z3b)*(-cosB*sinB+a*y1*y3b/(rb**3*cosB)+(rb*sinB-y1)/\
        rb*(2*(1-nu)*cosB-(rb*cosB+y3b)/(rb+z3b)*(1+a/(rb*cosB)))))

    v2cb2 = b2/4/np.pi/(1-nu)*(2*(1-nu)*(1-2*nu)*Fib*cotB**2+(1-2*nu)*y2/\
        (rb+y3b)*(-(1-2*nu-a/rb)*cotB+y1/(rb+y3b)*(nu+a/rb))-(1-2*nu)*\
        y2*cotB/(rb+z3b)*(1+a/(rb*cosB))-a*y2*(y3b-a)*cotB/rb**3+y2*\
        (y3b-a)/(rb*(rb+y3b))*((1-2*nu)*cotB-2*nu*y1/(rb+y3b)-a*y1/rb*\
        (1/rb+1/(rb+y3b)))+y2*(y3b-a)*cotB/(rb*(rb+z3b))*(-2*(1-nu)*\
        cosB+(rb*cosB+y3b)/(rb+z3b)*(1+a/(rb*cosB))+a*y3b/(rb**2*cosB)))

    v3cb2 = b2/4/np.pi/(1-nu)*(-2*(1-nu)*(1-2*nu)*cotB*(np.log(rb+y3b)-cosB*\
        np.log(rb+z3b))-2*(1-nu)*y1/(rb+y3b)*(2*nu+a/rb)+2*(1-nu)*z1b/(rb+\
        z3b)*(cosB+a/rb)+(y3b-a)/rb*((1-2*nu)*cotB-2*nu*y1/(rb+y3b)-a*\
        y1/rb**2)-(y3b-a)/(rb+z3b)*(cosB*sinB+(rb*cosB+y3b)*cotB/rb*\
        (2*(1-nu)*cosB-(rb*cosB+y3b)/(rb+z3b))+a/rb*(sinB-y3b*z1b/\
        rb**2-z1b*(rb*cosB+y3b)/(rb*(rb+z3b)))))

    v1cb3 = b3/4/np.pi/(1-nu)*((1-2*nu)*(y2/(rb+y3b)*(1+a/rb)-y2*cosB/(rb+\
        z3b)*(cosB+a/rb))-y2*(y3b-a)/rb*(a/rb**2+1/(rb+y3b))+y2*\
        (y3b-a)*cosB/(rb*(rb+z3b))*((rb*cosB+y3b)/(rb+z3b)*(cosB+a/\
        rb)+a*y3b/rb**2))

    v2cb3 = b3/4/np.pi/(1-nu)*((1-2*nu)*(-sinB*np.log(rb+z3b)-y1/(rb+y3b)*(1+a/\
        rb)+z1b/(rb+z3b)*(cosB+a/rb))+y1*(y3b-a)/rb*(a/rb**2+1/(rb+\
        y3b))-(y3b-a)/(rb+z3b)*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/\
        rb**2)-1/(rb*(rb+z3b))*(y2**2*cosB*sinB-a*z1b/rb*(rb*cosB+y3b))))

    v3cb3 = b3/4/np.pi/(1-nu)*(2*(1-nu)*Fib+2*(1-nu)*(y2*sinB/(rb+z3b)*(cosB+\
        a/rb))+y2*(y3b-a)*sinB/(rb*(rb+z3b))*(1+(rb*cosB+y3b)/(rb+\
        z3b)*(cosB+a/rb)+a*y3b/rb**2))

    return v1cb1+v1cb2+v1cb3, v2cb1+v2cb2+v2cb3, v3cb1+v3cb2+v3cb3

########################################################################
#################### Full-space strain functions #######################
########################################################################

def TDstrainFS(obs,tri,slip,nu):
    # Calculates strains associated with a triangular dislocation 
    # in an elastic full-space.
    
    # require ndmin=2 in case of only 1 obs point being passed
    if np.ndim(obs)<2:
        obs=np.array(obs,ndmin=2)
    
    # define slip vector
    slip_b = np.array([slip[2], slip[0], slip[1]])
    
    # convert coordinates from EFCS to TDCS
    transform,transformed_obs,transformed_tri,e12,e13,e23,A,B,C = setupTDCS(obs,tri)
    
    # select appropriate angular dislocations for artefact-free solution
    Ipos,Ineg,Inan = trimodefinder(np.array([transformed_obs[1],\
                                             transformed_obs[2],transformed_obs[0]]),\
                                             transformed_tri[:,1:])
    
    exx=np.empty(len(transformed_obs[0]))
    eyy=np.empty(len(transformed_obs[0]))
    ezz=np.empty(len(transformed_obs[0]))
    exy=np.empty(len(transformed_obs[0]))
    exz=np.empty(len(transformed_obs[0]))
    eyz=np.empty(len(transformed_obs[0]))
        
    # Calculate the strain tensor components in TDCS
    # Configuration I
    if len(Ipos)>0:
        # Calculate first angular dislocation contribution
        Exx1Tp,Eyy1Tp,Ezz1Tp,Exy1Tp,Exz1Tp,Eyz1Tp = \
            TDSetupS(transformed_obs[:,Ipos],A,slip_b,nu,transformed_tri[0],-e13)
        # Calculate second angular dislocation contribution
        Exx2Tp,Eyy2Tp,Ezz2Tp,Exy2Tp,Exz2Tp,Eyz2Tp = \
            TDSetupS(transformed_obs[:,Ipos],B,slip_b,nu,transformed_tri[1],e12)
        # Calculate third angular dislocation contribution
        Exx3Tp,Eyy3Tp,Ezz3Tp,Exy3Tp,Exz3Tp,Eyz3Tp = \
            TDSetupS(transformed_obs[:,Ipos],C,slip_b,nu,transformed_tri[2],e23)

        exx[Ipos] = Exx1Tp+Exx2Tp+Exx3Tp
        eyy[Ipos] = Eyy1Tp+Eyy2Tp+Eyy3Tp
        ezz[Ipos] = Ezz1Tp+Ezz2Tp+Ezz3Tp
        exy[Ipos] = Exy1Tp+Exy2Tp+Exy3Tp
        exz[Ipos] = Exz1Tp+Exz2Tp+Exz3Tp
        eyz[Ipos] = Eyz1Tp+Eyz2Tp+Eyz3Tp

    # Configuration II
    if len(Ineg)>0:
        # Calculate first angular dislocation contribution
        Exx1Tn,Eyy1Tn,Ezz1Tn,Exy1Tn,Exz1Tn,Eyz1Tn = \
            TDSetupS(transformed_obs[:,Ineg],A,slip_b,nu,transformed_tri[0],e13)
        # Calculate second angular dislocation contribution
        Exx2Tn,Eyy2Tn,Ezz2Tn,Exy2Tn,Exz2Tn,Eyz2Tn = \
            TDSetupS(transformed_obs[:,Ineg],B,slip_b,nu,transformed_tri[1],-e12)
        # Calculate third angular dislocation contribution
        Exx3Tn,Eyy3Tn,Ezz3Tn,Exy3Tn,Exz3Tn,Eyz3Tn = \
            TDSetupS(transformed_obs[:,Ineg],C,slip_b,nu,transformed_tri[2],-e23)
        
        exx[Ineg] = Exx1Tn+Exx2Tn+Exx3Tn
        eyy[Ineg] = Eyy1Tn+Eyy2Tn+Eyy3Tn
        ezz[Ineg] = Ezz1Tn+Ezz2Tn+Ezz3Tn
        exy[Ineg] = Exy1Tn+Exy2Tn+Exy3Tn
        exz[Ineg] = Exz1Tn+Exz2Tn+Exz3Tn
        eyz[Ineg] = Eyz1Tn+Eyz2Tn+Eyz3Tn

    # points located exactly on the dislocation edge
    if len(Inan)>0:
        exx[Inan] = np.nan
        eyy[Inan] = np.nan
        ezz[Inan] = np.nan
        exy[Inan] = np.nan
        exz[Inan] = np.nan
        eyz[Inan] = np.nan

    # Transform the strain tensor components from TDCS into EFCS
    Exx,Eyy,Ezz,Exy,Exz,Eyz = TensTrans(exx,eyy,ezz,exy,exz,eyz,transform.T)

    return np.array([Exx,Eyy,Ezz,Exy,Exz,Eyz]).T

def TDSetupS(obs,alpha,slip_b,nu,TriVertex,SideVec):
    # TDSetupS transforms coordinates of the calculation points as well as 
    # slip vector components from ADCS into TDCS. It then calculates the 
    # strains in ADCS and transforms them into TDCS.
    
    # Transform calculation points and slip vector components from TDCS into ADCS
    A,y1,z1,by1,bz1 = TDtransform_pts_slip(obs,slip_b,TriVertex,SideVec)

    # Calculate strains associated with an angular dislocation in ADCS
    exx,eyy,ezz,exy,exz,eyz = AngDisStrain(obs[0],y1,z1,-np.pi+alpha,slip_b[0],by1,bz1,nu)
    
    # Transform strains from ADCS into TDCS 
    # 3x3 Transformation matrix
    B = np.row_stack(([1, 0, 0],np.column_stack((np.zeros((2,1)),np.array(A,ndmin=2).T))))
    exx,eyy,ezz,exy,exz,eyz = TensTrans(exx,eyy,ezz,exy,exz,eyz,B)
    
    return exx,eyy,ezz,exy,exz,eyz
              
def AngDisStrain(x,y,z,alpha,bx,by,bz,nu):
    # AngDisStrain calculates the strains associated with an angular 
    # dislocation in an elastic full-space.

    sinA = np.sin(alpha)
    cosA = np.cos(alpha)
    eta = y*cosA-z*sinA 
    zeta = y*sinA+z*cosA 

    x2 = x**2 
    y2 = y**2 
    z2 = z**2 
    r2 = x2+y2+z2 
    r = np.sqrt(r2) 
    r3 = r*r2 
    rz = r*(r-z) 
    r2z2 = r2*(r-z)**2 
    r3z = r3*(r-z) 

    W = zeta-r 
    W2 = W**2 
    Wr = W*r 
    W2r = W2*r 
    Wr3 = W*r3 
    W2r2 = W2*r2 

    C = (r*cosA-z)/Wr 
    S = (r*sinA-y)/Wr 
    
    # Partial derivatives of the Burgers' function
    rFi_rx = (eta/r/(r-zeta)-y/r/(r-z))/4/np.pi 
    rFi_ry = (x/r/(r-z)-cosA*x/r/(r-zeta))/4/np.pi 
    rFi_rz = (sinA*x/r/(r-zeta))/4/np.pi 

    Exx = bx*(rFi_rx)+\
        bx/8/np.pi/(1-nu)*(eta/Wr+eta*x2/W2r2-eta*x2/Wr3+y/rz-\
        x2*y/r2z2-x2*y/r3z)-\
        by*x/8/np.pi/(1-nu)*(((2*nu+1)/Wr+x2/W2r2-x2/Wr3)*cosA+\
        (2*nu+1)/rz-x2/r2z2-x2/r3z)+\
        bz*x*sinA/8/np.pi/(1-nu)*((2*nu+1)/Wr+x2/W2r2-x2/Wr3) 

    Eyy = by*(rFi_ry)+\
        bx/8/np.pi/(1-nu)*((1/Wr+S**2-y2/Wr3)*eta+(2*nu+1)*y/rz-y**3/r2z2-\
        y**3/r3z-2*nu*cosA*S)-\
        by*x/8/np.pi/(1-nu)*(1/rz-y2/r2z2-y2/r3z+\
        (1/Wr+S**2-y2/Wr3)*cosA)+\
        bz*x*sinA/8/np.pi/(1-nu)*(1/Wr+S**2-y2/Wr3) 

    Ezz = bz*(rFi_rz)+\
        bx/8/np.pi/(1-nu)*(eta/W/r+eta*C**2-eta*z2/Wr3+y*z/r3+\
        2*nu*sinA*C)-\
        by*x/8/np.pi/(1-nu)*((1/Wr+C**2-z2/Wr3)*cosA+z/r3)+\
        bz*x*sinA/8/np.pi/(1-nu)*(1/Wr+C**2-z2/Wr3) 

    Exy = bx*(rFi_ry)/2+by*(rFi_rx)/2-\
        bx/8/np.pi/(1-nu)*(x*y2/r2z2-nu*x/rz+x*y2/r3z-nu*x*cosA/Wr+\
        eta*x*S/Wr+eta*x*y/Wr3)+\
        by/8/np.pi/(1-nu)*(x2*y/r2z2-nu*y/rz+x2*y/r3z+nu*cosA*S+\
        x2*y*cosA/Wr3+x2*cosA*S/Wr)-\
        bz*sinA/8/np.pi/(1-nu)*(nu*S+x2*S/Wr+x2*y/Wr3) 

    Exz = bx*(rFi_rz)/2+bz*(rFi_rx)/2-\
        bx/8/np.pi/(1-nu)*(-x*y/r3+nu*x*sinA/Wr+eta*x*C/Wr+\
        eta*x*z/Wr3)+\
        by/8/np.pi/(1-nu)*(-x2/r3+nu/r+nu*cosA*C+x2*z*cosA/Wr3+\
        x2*cosA*C/Wr)-\
        bz*sinA/8/np.pi/(1-nu)*(nu*C+x2*C/Wr+x2*z/Wr3) 

    Eyz = by*(rFi_rz)/2+bz*(rFi_ry)/2+\
        bx/8/np.pi/(1-nu)*(y2/r3-nu/r-nu*cosA*C+nu*sinA*S+eta*sinA*cosA/W2-\
        eta*(y*cosA+z*sinA)/W2r+eta*y*z/W2r2-eta*y*z/Wr3)-\
        by*x/8/np.pi/(1-nu)*(y/r3+sinA*cosA**2/W2-cosA*(y*cosA+z*sinA)/\
        W2r+y*z*cosA/W2r2-y*z*cosA/Wr3)-\
        bz*x*sinA/8/np.pi/(1-nu)*(y*z/Wr3-sinA*cosA/W2+(y*cosA+z*sinA)/\
        W2r-y*z/W2r2) 

    return Exx,Eyy,Ezz,Exy,Exz,Eyz

########################################################################
#################### Half-space strain functions #######################
########################################################################

def TDstrainHS(obs,tri,slip,nu):
    # Calculates stresses and strains associated with a triangular dislocation 
    # in an elastic half-space.
    
    # require ndmin=2 in case of only 1 obs point being passed
    if np.ndim(obs)<2:
        obs=np.array(obs,ndmin=2)

    assert all(obs[:,2]<=0), 'Half-space solution: observation Z coordinates must be zero or negative!'
    assert all(tri[:,2]<=0), 'Half-space solution: triangle Z coordinates must be zero or negative!'
    
    # Calculate main dislocation contribution to strains and stresses
    StrMS = TDstrainFS(obs, tri, slip, nu)

    # Calculate harmonic function contribution to strains and stresses
    StrFSC = TDstrain_HarFunc(obs, tri, slip, nu)

    # Calculate image dislocation contribution to strains and stresses
    tri_img = np.copy(tri) # do not modify tri!
    tri_img[:,2] = -tri_img[:,2]
    StrIS = TDstrainFS(obs, tri_img, slip, nu)

    if all(tri[:,2]==0):
        StrIS[:,4] = -StrIS[:,4]
        StrIS[:,5] = -StrIS[:,5]

    # Calculate the complete strain tensor components in EFCS
    Strain = StrMS+StrIS+StrFSC
    
    return Strain

def TDstrain_HarFunc(obs, tri, slip, nu):
    # TDstress_HarFunc calculates the harmonic function contribution to the
    # strains and stresses associated with a triangular dislocation in a 
    # half-space. The function cancels the surface normal tractions induced by 
    # the main and image dislocations.

    bx = slip[2] # Tensile-slip
    by = slip[0] # Strike-slip
    bz = slip[1] # Dip-slip

    # Transform slip vector components from TDCS into EFCS
    A = build_tri_coordinate_system(tri)

    bX,bY,bZ = CoordTrans(bx,by,bz,A.T)

    # Calculate contribution of angular dislocation pair on each TD side 
    Strain1 = AngSetupFSC_S(obs,bX,bY,bZ,tri[0],tri[1],nu) # P1P2
    Strain2 = AngSetupFSC_S(obs,bX,bY,bZ,tri[1],tri[2],nu) # P2P3
    Strain3 = AngSetupFSC_S(obs,bX,bY,bZ,tri[2],tri[0],nu) # P3P1

    # Calculate total harmonic function contribution to strains and stresses
    return Strain1+Strain2+Strain3

def AngSetupFSC_S(obs,bX,bY,bZ,PA,PB,nu):
    # AngSetupFSC_S calculates the Free Surface Correction to strains 
    # associated with angular dislocation pair on each TD side.
    
    npts=len(obs[:,0])
    
    # Calculate TD side vector and the angle of the angular dislocation pair
    SideVec = PB-PA
    eZ = np.array([0, 0, 1])
    beta = np.arccos(-SideVec.dot(eZ)/np.linalg.norm(SideVec))
    
    if (np.abs(beta) < np.finfo(float).eps or np.abs(np.pi-beta) < np.finfo(float).eps):
        Strain = np.zeros((npts,6))
    else:
        ey1 = normalize(np.array([SideVec[0],SideVec[1],0]))
        ey3 = -eZ
        ey2 = np.cross(ey3,ey1)
        A = np.array([ey1,ey2,ey3]) # Transformation matrix

        # Transform coordinates from EFCS to the first ADCS
        y1A,y2A,y3A = CoordTrans(obs[:,0]-PA[0],obs[:,1]-PA[1],obs[:,2]-PA[2],A)
        # Transform coordinates from EFCS to the second ADCS
        y1AB,y2AB,y3AB = CoordTrans(SideVec[0],SideVec[1],SideVec[2],A)
        y1B = y1A-y1AB
        y2B = y2A-y2AB
        y3B = y3A-y3AB

        # Transform slip vector components from EFCS to ADCS
        b1,b2,b3 = CoordTrans(bX,bY,bZ,A)

        # Determine the best arteact-free configuration for the calculation
        # points near the free furface
        Ipos = (beta*y1A >= 0)
        Ineg = np.logical_not(Ipos)
        
        # For singularities at surface
        v11A = np.empty(npts)
        v22A = np.empty(npts)
        v33A = np.empty(npts)
        v12A = np.empty(npts)
        v13A = np.empty(npts)
        v23A = np.empty(npts)

        v11B = np.empty(npts)
        v22B = np.empty(npts)
        v33B = np.empty(npts)
        v12B = np.empty(npts)
        v13B = np.empty(npts)
        v23B = np.empty(npts)
        
        # Configuration I
        [v11A[Ipos],v22A[Ipos],v33A[Ipos],v12A[Ipos],v13A[Ipos],v23A[Ipos]] = \
            AngDisStrainFSC(-y1A[Ipos],-y2A[Ipos],y3A[Ipos],\
            np.pi-beta,-b1,-b2,b3,nu,-PA[2])
        v13A[Ipos] = -v13A[Ipos]
        v23A[Ipos] = -v23A[Ipos]

        [v11B[Ipos],v22B[Ipos],v33B[Ipos],v12B[Ipos],v13B[Ipos],v23B[Ipos]] = \
            AngDisStrainFSC(-y1B[Ipos],-y2B[Ipos],y3B[Ipos],\
            np.pi-beta,-b1,-b2,b3,nu,-PB[2])
        v13B[Ipos] = -v13B[Ipos]
        v23B[Ipos] = -v23B[Ipos]

        # Configuration II
        [v11A[Ineg],v22A[Ineg],v33A[Ineg],v12A[Ineg],v13A[Ineg],v23A[Ineg]] = \
            AngDisStrainFSC(y1A[Ineg],y2A[Ineg],y3A[Ineg],\
            beta,b1,b2,b3,nu,-PA[2])

        [v11B[Ineg],v22B[Ineg],v33B[Ineg],v12B[Ineg],v13B[Ineg],v23B[Ineg]] = \
            AngDisStrainFSC(y1B[Ineg],y2B[Ineg],y3B[Ineg],\
            beta,b1,b2,b3,nu,-PB[2])
        
        # Calculate total Free Surface Correction to strains in ADCS
        v11 = v11B-v11A
        v22 = v22B-v22A
        v33 = v33B-v33A
        v12 = v12B-v12A
        v13 = v13B-v13A
        v23 = v23B-v23A

        # Transform total Free Surface Correction to strains from ADCS to EFCS
        Exx,Eyy,Ezz,Exy,Exz,Eyz = TensTrans(v11,v22,v33,v12,v13,v23,A.T)
        Strain = np.asarray([Exx,Eyy,Ezz,Exy,Exz,Eyz]).T
                
    return Strain
                                              
def AngDisStrainFSC(y1,y2,y3,beta,b1,b2,b3,nu,a):
    # AngDisStrainFSC calculates the harmonic function contribution to the 
    # strains associated with an angular dislocation in an elastic half-space.

    sinB = np.sin(beta)
    cosB = np.cos(beta)
    cotB = 1.0/np.tan(beta)
    y3b = y3+2*a
    z1b = y1*cosB+y3b*sinB
    z3b = -y1*sinB+y3b*cosB
    rb2 = y1**2+y2**2+y3b**2
    rb = np.sqrt(rb2)

    W1 = rb*cosB+y3b
    W2 = cosB+a/rb
    W3 = cosB+y3b/rb
    W4 = nu+a/rb
    W5 = 2*nu+a/rb
    W6 = rb+y3b
    W7 = rb+z3b
    W8 = y3+a
    W9 = 1+a/rb/cosB

    N1 = 1-2*nu

    # Partial derivatives of the Burgers' function
    rFib_ry2 = z1b/rb/(rb+z3b)-y1/rb/(rb+y3b) # y2 = x in ADCS
    rFib_ry1 = y2/rb/(rb+y3b)-cosB*y2/rb/(rb+z3b) # y1 =y in ADCS
    rFib_ry3 = -sinB*y2/rb/(rb+z3b) # y3 = z in ADCS

    v11 = b1*(1/4*((-2+2*nu)*N1*rFib_ry1*cotB**2-N1*y2/W6**2*((1-W5)*cotB-\
        y1/W6*W4)/rb*y1+N1*y2/W6*(a/rb**3*y1*cotB-1/W6*W4+y1**2/\
        W6**2*W4/rb+y1**2/W6*a/rb**3)-N1*y2*cosB*cotB/W7**2*W2*(y1/\
        rb-sinB)-N1*y2*cosB*cotB/W7*a/rb**3*y1-3*a*y2*W8*cotB/rb**5*\
        y1-y2*W8/rb**3/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1-y2*W8/\
        rb2/W6**2*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1+y2*W8/rb/W6*\
        (1/W6*W5-y1**2/W6**2*W5/rb-y1**2/W6*a/rb**3+a/rb2-2*a*y1**\
        2/rb2**2)-y2*W8/rb**3/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+\
        (2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)*y1-y2*W8/rb/\
        W7**2*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*\
        cosB)-a*y3b*cosB*cotB/rb2)*(y1/rb-sinB)+y2*W8/rb/W7*(-cosB/\
        W7**2*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)*(y1/\
        rb-sinB)+cosB/W7*(1/rb*cosB*y1*(N1*cosB-a/rb)*cotB+W1*a/rb**\
        3*y1*cotB+(2-2*nu)*(1/rb*sinB*y1-1)*cosB)+2*a*y3b*cosB*cotB/\
        rb2**2*y1))/np.pi/(1-nu))+\
        b2*(1/4*(N1*(((2-2*nu)*cotB**2+nu)/rb*y1/W6-((2-2*nu)*cotB**2+1)*\
        cosB*(y1/rb-sinB)/W7)-N1/W6**2*(-N1*y1*cotB+nu*y3b-a+a*y1*\
        cotB/rb+y1**2/W6*W4)/rb*y1+N1/W6*(-N1*cotB+a*cotB/rb-a*\
        y1**2*cotB/rb**3+2*y1/W6*W4-y1**3/W6**2*W4/rb-y1**3/W6*a/\
        rb**3)+N1*cotB/W7**2*(z1b*cosB-a*(rb*sinB-y1)/rb/cosB)*(y1/\
        rb-sinB)-N1*cotB/W7*(cosB**2-a*(1/rb*sinB*y1-1)/rb/cosB+a*\
        (rb*sinB-y1)/rb**3/cosB*y1)-a*W8*cotB/rb**3+3*a*y1**2*W8*\
        cotB/rb**5-W8/W6**2*(2*nu+1/rb*(N1*y1*cotB+a)-y1**2/rb/W6*\
        W5-a*y1**2/rb**3)/rb*y1+W8/W6*(-1/rb**3*(N1*y1*cotB+a)*y1+\
        1/rb*N1*cotB-2*y1/rb/W6*W5+y1**3/rb**3/W6*W5+y1**3/rb2/\
        W6**2*W5+y1**3/rb2**2/W6*a-2*a/rb**3*y1+3*a*y1**3/rb**5)-W8*\
        cotB/W7**2*(-cosB*sinB+a*y1*y3b/rb**3/cosB+(rb*sinB-y1)/rb*\
        ((2-2*nu)*cosB-W1/W7*W9))*(y1/rb-sinB)+W8*cotB/W7*(a*y3b/\
        rb**3/cosB-3*a*y1**2*y3b/rb**5/cosB+(1/rb*sinB*y1-1)/rb*\
        ((2-2*nu)*cosB-W1/W7*W9)-(rb*sinB-y1)/rb**3*((2-2*nu)*cosB-W1/\
        W7*W9)*y1+(rb*sinB-y1)/rb*(-1/rb*cosB*y1/W7*W9+W1/W7**2*\
        W9*(y1/rb-sinB)+W1/W7*a/rb**3/cosB*y1)))/np.pi/(1-nu))+\
        b3*(1/4*(N1*(-y2/W6**2*(1+a/rb)/rb*y1-y2/W6*a/rb**3*y1+y2*\
        cosB/W7**2*W2*(y1/rb-sinB)+y2*cosB/W7*a/rb**3*y1)+y2*W8/\
        rb**3*(a/rb2+1/W6)*y1-y2*W8/rb*(-2*a/rb2**2*y1-1/W6**2/\
        rb*y1)-y2*W8*cosB/rb**3/W7*(W1/W7*W2+a*y3b/rb2)*y1-y2*W8*\
        cosB/rb/W7**2*(W1/W7*W2+a*y3b/rb2)*(y1/rb-sinB)+y2*W8*\
        cosB/rb/W7*(1/rb*cosB*y1/W7*W2-W1/W7**2*W2*(y1/rb-sinB)-\
        W1/W7*a/rb**3*y1-2*a*y3b/rb2**2*y1))/np.pi/(1-nu))

    v22 = b1*(1/4*(N1*(((2-2*nu)*cotB**2-nu)/rb*y2/W6-((2-2*nu)*cotB**2+1-\
        2*nu)*cosB/rb*y2/W7)+N1/W6**2*(y1*cotB*(1-W5)+nu*y3b-a+y2**\
        2/W6*W4)/rb*y2-N1/W6*(a*y1*cotB/rb**3*y2+2*y2/W6*W4-y2**\
        3/W6**2*W4/rb-y2**3/W6*a/rb**3)+N1*z1b*cotB/W7**2*W2/rb*\
        y2+N1*z1b*cotB/W7*a/rb**3*y2+3*a*y2*W8*cotB/rb**5*y1-W8/\
        W6**2*(-2*nu+1/rb*(N1*y1*cotB-a)+y2**2/rb/W6*W5+a*y2**2/\
        rb**3)/rb*y2+W8/W6*(-1/rb**3*(N1*y1*cotB-a)*y2+2*y2/rb/\
        W6*W5-y2**3/rb**3/W6*W5-y2**3/rb2/W6**2*W5-y2**3/rb2**2/W6*\
        a+2*a/rb**3*y2-3*a*y2**3/rb**5)-W8/W7**2*(cosB**2-1/rb*(N1*\
        z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**3-1/rb/W7*(y2**2*cosB**2-\
        a*z1b*cotB/rb*W1))/rb*y2+W8/W7*(1/rb**3*(N1*z1b*cotB+a*\
        cosB)*y2-3*a*y3b*z1b*cotB/rb**5*y2+1/rb**3/W7*(y2**2*cosB**2-\
        a*z1b*cotB/rb*W1)*y2+1/rb2/W7**2*(y2**2*cosB**2-a*z1b*cotB/\
        rb*W1)*y2-1/rb/W7*(2*y2*cosB**2+a*z1b*cotB/rb**3*W1*y2-a*\
        z1b*cotB/rb2*cosB*y2)))/np.pi/(1-nu))+\
        b2*(1/4*((2-2*nu)*N1*rFib_ry2*cotB**2+N1/W6*((W5-1)*cotB+y1/W6*\
        W4)-N1*y2**2/W6**2*((W5-1)*cotB+y1/W6*W4)/rb+N1*y2/W6*(-a/\
        rb**3*y2*cotB-y1/W6**2*W4/rb*y2-y2/W6*a/rb**3*y1)-N1*cotB/\
        W7*W9+N1*y2**2*cotB/W7**2*W9/rb+N1*y2**2*cotB/W7*a/rb**3/\
        cosB-a*W8*cotB/rb**3+3*a*y2**2*W8*cotB/rb**5+W8/rb/W6*(N1*\
        cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))-y2**2*W8/rb**3/W6*\
        (N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))-y2**2*W8/rb2/W6**\
        2*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))+y2*W8/rb/W6*\
        (2*nu*y1/W6**2/rb*y2+a*y1/rb**3*(1/rb+1/W6)*y2-a*y1/rb*\
        (-1/rb**3*y2-1/W6**2/rb*y2))+W8*cotB/rb/W7*((-2+2*nu)*cosB+\
        W1/W7*W9+a*y3b/rb2/cosB)-y2**2*W8*cotB/rb**3/W7*((-2+2*nu)*\
        cosB+W1/W7*W9+a*y3b/rb2/cosB)-y2**2*W8*cotB/rb2/W7**2*((-2+\
        2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)+y2*W8*cotB/rb/W7*(1/\
        rb*cosB*y2/W7*W9-W1/W7**2*W9/rb*y2-W1/W7*a/rb**3/cosB*y2-\
        2*a*y3b/rb2**2/cosB*y2))/np.pi/(1-nu))+\
        b3*(1/4*(N1*(-sinB/rb*y2/W7+y2/W6**2*(1+a/rb)/rb*y1+y2/W6*\
        a/rb**3*y1-z1b/W7**2*W2/rb*y2-z1b/W7*a/rb**3*y2)-y2*W8/\
        rb**3*(a/rb2+1/W6)*y1+y1*W8/rb*(-2*a/rb2**2*y2-1/W6**2/\
        rb*y2)+W8/W7**2*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1/\
        rb/W7*(y2**2*cosB*sinB-a*z1b/rb*W1))/rb*y2-W8/W7*(sinB*a/\
        rb**3*y2-z1b/rb**3*(1+a*y3b/rb2)*y2-2*z1b/rb**5*a*y3b*y2+\
        1/rb**3/W7*(y2**2*cosB*sinB-a*z1b/rb*W1)*y2+1/rb2/W7**2*\
        (y2**2*cosB*sinB-a*z1b/rb*W1)*y2-1/rb/W7*(2*y2*cosB*sinB+a*\
        z1b/rb**3*W1*y2-a*z1b/rb2*cosB*y2)))/np.pi/(1-nu))

    v33 = b1*(1/4*((2-2*nu)*(N1*rFib_ry3*cotB-y2/W6**2*W5*(y3b/rb+1)-\
        1/2*y2/W6*a/rb**3*2*y3b+y2*cosB/W7**2*W2*W3+1/2*y2*cosB/W7*\
        a/rb**3*2*y3b)+y2/rb*(2*nu/W6+a/rb2)-1/2*y2*W8/rb**3*(2*\
        nu/W6+a/rb2)*2*y3b+y2*W8/rb*(-2*nu/W6**2*(y3b/rb+1)-a/\
        rb2**2*2*y3b)+y2*cosB/rb/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-\
        1/2*y2*W8*cosB/rb**3/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)*2*\
        y3b-y2*W8*cosB/rb/W7**2*(1-2*nu-W1/W7*W2-a*y3b/rb2)*W3+y2*\
        W8*cosB/rb/W7*(-(cosB*y3b/rb+1)/W7*W2+W1/W7**2*W2*W3+1/2*\
        W1/W7*a/rb**3*2*y3b-a/rb2+a*y3b/rb2**2*2*y3b))/np.pi/(1-nu))+\
        b2*(1/4*((-2+2*nu)*N1*cotB*((y3b/rb+1)/W6-cosB*W3/W7)+(2-2*nu)*\
        y1/W6**2*W5*(y3b/rb+1)+1/2*(2-2*nu)*y1/W6*a/rb**3*2*y3b+(2-\
        2*nu)*sinB/W7*W2-(2-2*nu)*z1b/W7**2*W2*W3-1/2*(2-2*nu)*z1b/\
        W7*a/rb**3*2*y3b+1/rb*(N1*cotB-2*nu*y1/W6-a*y1/rb2)-1/2*\
        W8/rb**3*(N1*cotB-2*nu*y1/W6-a*y1/rb2)*2*y3b+W8/rb*(2*nu*\
        y1/W6**2*(y3b/rb+1)+a*y1/rb2**2*2*y3b)-1/W7*(cosB*sinB+W1*\
        cotB/rb*((2-2*nu)*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*\
        W1/rb/W7))+W8/W7**2*(cosB*sinB+W1*cotB/rb*((2-2*nu)*cosB-W1/\
        W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*W3-W8/W7*((cosB*\
        y3b/rb+1)*cotB/rb*((2-2*nu)*cosB-W1/W7)-1/2*W1*cotB/rb**3*\
        ((2-2*nu)*cosB-W1/W7)*2*y3b+W1*cotB/rb*(-(cosB*y3b/rb+1)/W7+\
        W1/W7**2*W3)-1/2*a/rb**3*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7)*\
        2*y3b+a/rb*(-z1b/rb2-y3b*sinB/rb2+y3b*z1b/rb2**2*2*y3b-\
        sinB*W1/rb/W7-z1b*(cosB*y3b/rb+1)/rb/W7+1/2*z1b*W1/rb**3/\
        W7*2*y3b+z1b*W1/rb/W7**2*W3)))/np.pi/(1-nu))+\
        b3*(1/4*((2-2*nu)*rFib_ry3-(2-2*nu)*y2*sinB/W7**2*W2*W3-1/2*\
        (2-2*nu)*y2*sinB/W7*a/rb**3*2*y3b+y2*sinB/rb/W7*(1+W1/W7*\
        W2+a*y3b/rb2)-1/2*y2*W8*sinB/rb**3/W7*(1+W1/W7*W2+a*y3b/\
        rb2)*2*y3b-y2*W8*sinB/rb/W7**2*(1+W1/W7*W2+a*y3b/rb2)*W3+\
        y2*W8*sinB/rb/W7*((cosB*y3b/rb+1)/W7*W2-W1/W7**2*W2*W3-\
        1/2*W1/W7*a/rb**3*2*y3b+a/rb2-a*y3b/rb2**2*2*y3b))/np.pi/(1-nu))

    v12 = b1/2*(1/4*((-2+2*nu)*N1*rFib_ry2*cotB**2+N1/W6*((1-W5)*cotB-y1/\
        W6*W4)-N1*y2**2/W6**2*((1-W5)*cotB-y1/W6*W4)/rb+N1*y2/W6*\
        (a/rb**3*y2*cotB+y1/W6**2*W4/rb*y2+y2/W6*a/rb**3*y1)+N1*\
        cosB*cotB/W7*W2-N1*y2**2*cosB*cotB/W7**2*W2/rb-N1*y2**2*cosB*\
        cotB/W7*a/rb**3+a*W8*cotB/rb**3-3*a*y2**2*W8*cotB/rb**5+W8/\
        rb/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)-y2**2*W8/rb**3/W6*(-N1*\
        cotB+y1/W6*W5+a*y1/rb2)-y2**2*W8/rb2/W6**2*(-N1*cotB+y1/\
        W6*W5+a*y1/rb2)+y2*W8/rb/W6*(-y1/W6**2*W5/rb*y2-y2/W6*\
        a/rb**3*y1-2*a*y1/rb2**2*y2)+W8/rb/W7*(cosB/W7*(W1*(N1*\
        cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/\
        rb2)-y2**2*W8/rb**3/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-\
        2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)-y2**2*W8/rb2/\
        W7**2*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*\
        cosB)-a*y3b*cosB*cotB/rb2)+y2*W8/rb/W7*(-cosB/W7**2*(W1*\
        (N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)/rb*y2+cosB/\
        W7*(1/rb*cosB*y2*(N1*cosB-a/rb)*cotB+W1*a/rb**3*y2*cotB+(2-2*\
        nu)/rb*sinB*y2*cosB)+2*a*y3b*cosB*cotB/rb2**2*y2))/np.pi/(1-nu))+\
        b2/2*(1/4*(N1*(((2-2*nu)*cotB**2+nu)/rb*y2/W6-((2-2*nu)*cotB**2+1)*\
        cosB/rb*y2/W7)-N1/W6**2*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/rb+\
        y1**2/W6*W4)/rb*y2+N1/W6*(-a*y1*cotB/rb**3*y2-y1**2/W6**\
        2*W4/rb*y2-y1**2/W6*a/rb**3*y2)+N1*cotB/W7**2*(z1b*cosB-a*\
        (rb*sinB-y1)/rb/cosB)/rb*y2-N1*cotB/W7*(-a/rb2*sinB*y2/\
        cosB+a*(rb*sinB-y1)/rb**3/cosB*y2)+3*a*y2*W8*cotB/rb**5*y1-\
        W8/W6**2*(2*nu+1/rb*(N1*y1*cotB+a)-y1**2/rb/W6*W5-a*y1**2/\
        rb**3)/rb*y2+W8/W6*(-1/rb**3*(N1*y1*cotB+a)*y2+y1**2/rb**\
        3/W6*W5*y2+y1**2/rb2/W6**2*W5*y2+y1**2/rb2**2/W6*a*y2+3*\
        a*y1**2/rb**5*y2)-W8*cotB/W7**2*(-cosB*sinB+a*y1*y3b/rb**3/\
        cosB+(rb*sinB-y1)/rb*((2-2*nu)*cosB-W1/W7*W9))/rb*y2+W8*cotB/\
        W7*(-3*a*y1*y3b/rb**5/cosB*y2+1/rb2*sinB*y2*((2-2*nu)*cosB-\
        W1/W7*W9)-(rb*sinB-y1)/rb**3*((2-2*nu)*cosB-W1/W7*W9)*y2+(rb*\
        sinB-y1)/rb*(-1/rb*cosB*y2/W7*W9+W1/W7**2*W9/rb*y2+W1/W7*\
        a/rb**3/cosB*y2)))/np.pi/(1-nu))+\
        b3/2*(1/4*(N1*(1/W6*(1+a/rb)-y2**2/W6**2*(1+a/rb)/rb-y2**2/\
        W6*a/rb**3-cosB/W7*W2+y2**2*cosB/W7**2*W2/rb+y2**2*cosB/W7*\
        a/rb**3)-W8/rb*(a/rb2+1/W6)+y2**2*W8/rb**3*(a/rb2+1/W6)-\
        y2*W8/rb*(-2*a/rb2**2*y2-1/W6**2/rb*y2)+W8*cosB/rb/W7*\
        (W1/W7*W2+a*y3b/rb2)-y2**2*W8*cosB/rb**3/W7*(W1/W7*W2+a*\
        y3b/rb2)-y2**2*W8*cosB/rb2/W7**2*(W1/W7*W2+a*y3b/rb2)+y2*\
        W8*cosB/rb/W7*(1/rb*cosB*y2/W7*W2-W1/W7**2*W2/rb*y2-W1/\
        W7*a/rb**3*y2-2*a*y3b/rb2**2*y2))/np.pi/(1-nu))+\
        b1/2*(1/4*(N1*(((2-2*nu)*cotB**2-nu)/rb*y1/W6-((2-2*nu)*cotB**2+1-\
        2*nu)*cosB*(y1/rb-sinB)/W7)+N1/W6**2*(y1*cotB*(1-W5)+nu*y3b-\
        a+y2**2/W6*W4)/rb*y1-N1/W6*((1-W5)*cotB+a*y1**2*cotB/rb**3-\
        y2**2/W6**2*W4/rb*y1-y2**2/W6*a/rb**3*y1)-N1*cosB*cotB/W7*\
        W2+N1*z1b*cotB/W7**2*W2*(y1/rb-sinB)+N1*z1b*cotB/W7*a/rb**\
        3*y1-a*W8*cotB/rb**3+3*a*y1**2*W8*cotB/rb**5-W8/W6**2*(-2*\
        nu+1/rb*(N1*y1*cotB-a)+y2**2/rb/W6*W5+a*y2**2/rb**3)/rb*\
        y1+W8/W6*(-1/rb**3*(N1*y1*cotB-a)*y1+1/rb*N1*cotB-y2**2/\
        rb**3/W6*W5*y1-y2**2/rb2/W6**2*W5*y1-y2**2/rb2**2/W6*a*y1-\
        3*a*y2**2/rb**5*y1)-W8/W7**2*(cosB**2-1/rb*(N1*z1b*cotB+a*\
        cosB)+a*y3b*z1b*cotB/rb**3-1/rb/W7*(y2**2*cosB**2-a*z1b*cotB/\
        rb*W1))*(y1/rb-sinB)+W8/W7*(1/rb**3*(N1*z1b*cotB+a*cosB)*\
        y1-1/rb*N1*cosB*cotB+a*y3b*cosB*cotB/rb**3-3*a*y3b*z1b*cotB/\
        rb**5*y1+1/rb**3/W7*(y2**2*cosB**2-a*z1b*cotB/rb*W1)*y1+1/\
        rb/W7**2*(y2**2*cosB**2-a*z1b*cotB/rb*W1)*(y1/rb-sinB)-1/rb/\
        W7*(-a*cosB*cotB/rb*W1+a*z1b*cotB/rb**3*W1*y1-a*z1b*cotB/\
        rb2*cosB*y1)))/np.pi/(1-nu))+\
        b2/2*(1/4*((2-2*nu)*N1*rFib_ry1*cotB**2-N1*y2/W6**2*((W5-1)*cotB+\
        y1/W6*W4)/rb*y1+N1*y2/W6*(-a/rb**3*y1*cotB+1/W6*W4-y1**\
        2/W6**2*W4/rb-y1**2/W6*a/rb**3)+N1*y2*cotB/W7**2*W9*(y1/\
        rb-sinB)+N1*y2*cotB/W7*a/rb**3/cosB*y1+3*a*y2*W8*cotB/rb**\
        5*y1-y2*W8/rb**3/W6*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/\
        W6))*y1-y2*W8/rb2/W6**2*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/\
        rb+1/W6))*y1+y2*W8/rb/W6*(-2*nu/W6+2*nu*y1**2/W6**2/rb-a/\
        rb*(1/rb+1/W6)+a*y1**2/rb**3*(1/rb+1/W6)-a*y1/rb*(-1/\
        rb**3*y1-1/W6**2/rb*y1))-y2*W8*cotB/rb**3/W7*((-2+2*nu)*\
        cosB+W1/W7*W9+a*y3b/rb2/cosB)*y1-y2*W8*cotB/rb/W7**2*((-2+\
        2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)*(y1/rb-sinB)+y2*W8*\
        cotB/rb/W7*(1/rb*cosB*y1/W7*W9-W1/W7**2*W9*(y1/rb-sinB)-\
        W1/W7*a/rb**3/cosB*y1-2*a*y3b/rb2**2/cosB*y1))/np.pi/(1-nu))+\
        b3/2*(1/4*(N1*(-sinB*(y1/rb-sinB)/W7-1/W6*(1+a/rb)+y1**2/W6**\
        2*(1+a/rb)/rb+y1**2/W6*a/rb**3+cosB/W7*W2-z1b/W7**2*W2*\
        (y1/rb-sinB)-z1b/W7*a/rb**3*y1)+W8/rb*(a/rb2+1/W6)-y1**2*\
        W8/rb**3*(a/rb2+1/W6)+y1*W8/rb*(-2*a/rb2**2*y1-1/W6**2/\
        rb*y1)+W8/W7**2*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1/\
        rb/W7*(y2**2*cosB*sinB-a*z1b/rb*W1))*(y1/rb-sinB)-W8/W7*\
        (sinB*a/rb**3*y1+cosB/rb*(1+a*y3b/rb2)-z1b/rb**3*(1+a*y3b/\
        rb2)*y1-2*z1b/rb**5*a*y3b*y1+1/rb**3/W7*(y2**2*cosB*sinB-a*\
        z1b/rb*W1)*y1+1/rb/W7**2*(y2**2*cosB*sinB-a*z1b/rb*W1)*\
        (y1/rb-sinB)-1/rb/W7*(-a*cosB/rb*W1+a*z1b/rb**3*W1*y1-a*\
        z1b/rb2*cosB*y1)))/np.pi/(1-nu))

    v13 = b1/2*(1/4*((-2+2*nu)*N1*rFib_ry3*cotB**2-N1*y2/W6**2*((1-W5)*\
        cotB-y1/W6*W4)*(y3b/rb+1)+N1*y2/W6*(1/2*a/rb**3*2*y3b*cotB+\
        y1/W6**2*W4*(y3b/rb+1)+1/2*y1/W6*a/rb**3*2*y3b)-N1*y2*cosB*\
        cotB/W7**2*W2*W3-1/2*N1*y2*cosB*cotB/W7*a/rb**3*2*y3b+a/\
        rb**3*y2*cotB-3/2*a*y2*W8*cotB/rb**5*2*y3b+y2/rb/W6*(-N1*\
        cotB+y1/W6*W5+a*y1/rb2)-1/2*y2*W8/rb**3/W6*(-N1*cotB+y1/\
        W6*W5+a*y1/rb2)*2*y3b-y2*W8/rb/W6**2*(-N1*cotB+y1/W6*W5+\
        a*y1/rb2)*(y3b/rb+1)+y2*W8/rb/W6*(-y1/W6**2*W5*(y3b/rb+\
        1)-1/2*y1/W6*a/rb**3*2*y3b-a*y1/rb2**2*2*y3b)+y2/rb/W7*\
        (cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-\
        a*y3b*cosB*cotB/rb2)-1/2*y2*W8/rb**3/W7*(cosB/W7*(W1*(N1*\
        cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/\
        rb2)*2*y3b-y2*W8/rb/W7**2*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+\
        (2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)*W3+y2*W8/rb/\
        W7*(-cosB/W7**2*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*\
        cosB)*W3+cosB/W7*((cosB*y3b/rb+1)*(N1*cosB-a/rb)*cotB+1/2*W1*\
        a/rb**3*2*y3b*cotB+1/2*(2-2*nu)/rb*sinB*2*y3b*cosB)-a*cosB*\
        cotB/rb2+a*y3b*cosB*cotB/rb2**2*2*y3b))/np.pi/(1-nu))+\
        b2/2*(1/4*(N1*(((2-2*nu)*cotB**2+nu)*(y3b/rb+1)/W6-((2-2*nu)*cotB**\
        2+1)*cosB*W3/W7)-N1/W6**2*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/\
        rb+y1**2/W6*W4)*(y3b/rb+1)+N1/W6*(nu-1/2*a*y1*cotB/rb**3*2*\
        y3b-y1**2/W6**2*W4*(y3b/rb+1)-1/2*y1**2/W6*a/rb**3*2*y3b)+\
        N1*cotB/W7**2*(z1b*cosB-a*(rb*sinB-y1)/rb/cosB)*W3-N1*cotB/\
        W7*(cosB*sinB-1/2*a/rb2*sinB*2*y3b/cosB+1/2*a*(rb*sinB-y1)/\
        rb**3/cosB*2*y3b)-a/rb**3*y1*cotB+3/2*a*y1*W8*cotB/rb**5*2*\
        y3b+1/W6*(2*nu+1/rb*(N1*y1*cotB+a)-y1**2/rb/W6*W5-a*y1**2/\
        rb**3)-W8/W6**2*(2*nu+1/rb*(N1*y1*cotB+a)-y1**2/rb/W6*W5-a*\
        y1**2/rb**3)*(y3b/rb+1)+W8/W6*(-1/2/rb**3*(N1*y1*cotB+a)*2*\
        y3b+1/2*y1**2/rb**3/W6*W5*2*y3b+y1**2/rb/W6**2*W5*(y3b/rb+\
        1)+1/2*y1**2/rb2**2/W6*a*2*y3b+3/2*a*y1**2/rb**5*2*y3b)+\
        cotB/W7*(-cosB*sinB+a*y1*y3b/rb**3/cosB+(rb*sinB-y1)/rb*((2-\
        2*nu)*cosB-W1/W7*W9))-W8*cotB/W7**2*(-cosB*sinB+a*y1*y3b/rb**\
        3/cosB+(rb*sinB-y1)/rb*((2-2*nu)*cosB-W1/W7*W9))*W3+W8*cotB/\
        W7*(a/rb**3/cosB*y1-3/2*a*y1*y3b/rb**5/cosB*2*y3b+1/2/\
        rb2*sinB*2*y3b*((2-2*nu)*cosB-W1/W7*W9)-1/2*(rb*sinB-y1)/rb**\
        3*((2-2*nu)*cosB-W1/W7*W9)*2*y3b+(rb*sinB-y1)/rb*(-(cosB*y3b/\
        rb+1)/W7*W9+W1/W7**2*W9*W3+1/2*W1/W7*a/rb**3/cosB*2*\
        y3b)))/np.pi/(1-nu))+\
        b3/2*(1/4*(N1*(-y2/W6**2*(1+a/rb)*(y3b/rb+1)-1/2*y2/W6*a/\
        rb**3*2*y3b+y2*cosB/W7**2*W2*W3+1/2*y2*cosB/W7*a/rb**3*2*\
        y3b)-y2/rb*(a/rb2+1/W6)+1/2*y2*W8/rb**3*(a/rb2+1/W6)*2*\
        y3b-y2*W8/rb*(-a/rb2**2*2*y3b-1/W6**2*(y3b/rb+1))+y2*cosB/\
        rb/W7*(W1/W7*W2+a*y3b/rb2)-1/2*y2*W8*cosB/rb**3/W7*(W1/\
        W7*W2+a*y3b/rb2)*2*y3b-y2*W8*cosB/rb/W7**2*(W1/W7*W2+a*\
        y3b/rb2)*W3+y2*W8*cosB/rb/W7*((cosB*y3b/rb+1)/W7*W2-W1/\
        W7**2*W2*W3-1/2*W1/W7*a/rb**3*2*y3b+a/rb2-a*y3b/rb2**2*2*\
        y3b))/np.pi/(1-nu))+\
        b1/2*(1/4*((2-2*nu)*(N1*rFib_ry1*cotB-y1/W6**2*W5/rb*y2-y2/W6*\
        a/rb**3*y1+y2*cosB/W7**2*W2*(y1/rb-sinB)+y2*cosB/W7*a/rb**\
        3*y1)-y2*W8/rb**3*(2*nu/W6+a/rb2)*y1+y2*W8/rb*(-2*nu/W6**\
        2/rb*y1-2*a/rb2**2*y1)-y2*W8*cosB/rb**3/W7*(1-2*nu-W1/W7*\
        W2-a*y3b/rb2)*y1-y2*W8*cosB/rb/W7**2*(1-2*nu-W1/W7*W2-a*\
        y3b/rb2)*(y1/rb-sinB)+y2*W8*cosB/rb/W7*(-1/rb*cosB*y1/W7*\
        W2+W1/W7**2*W2*(y1/rb-sinB)+W1/W7*a/rb**3*y1+2*a*y3b/rb2**\
        2*y1))/np.pi/(1-nu))+\
        b2/2*(1/4*((-2+2*nu)*N1*cotB*(1/rb*y1/W6-cosB*(y1/rb-sinB)/W7)-\
        (2-2*nu)/W6*W5+(2-2*nu)*y1**2/W6**2*W5/rb+(2-2*nu)*y1**2/W6*\
        a/rb**3+(2-2*nu)*cosB/W7*W2-(2-2*nu)*z1b/W7**2*W2*(y1/rb-\
        sinB)-(2-2*nu)*z1b/W7*a/rb**3*y1-W8/rb**3*(N1*cotB-2*nu*y1/\
        W6-a*y1/rb2)*y1+W8/rb*(-2*nu/W6+2*nu*y1**2/W6**2/rb-a/rb2+\
        2*a*y1**2/rb2**2)+W8/W7**2*(cosB*sinB+W1*cotB/rb*((2-2*nu)*\
        cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*(y1/rb-\
        sinB)-W8/W7*(1/rb2*cosB*y1*cotB*((2-2*nu)*cosB-W1/W7)-W1*\
        cotB/rb**3*((2-2*nu)*cosB-W1/W7)*y1+W1*cotB/rb*(-1/rb*cosB*\
        y1/W7+W1/W7**2*(y1/rb-sinB))-a/rb**3*(sinB-y3b*z1b/rb2-\
        z1b*W1/rb/W7)*y1+a/rb*(-y3b*cosB/rb2+2*y3b*z1b/rb2**2*y1-\
        cosB*W1/rb/W7-z1b/rb2*cosB*y1/W7+z1b*W1/rb**3/W7*y1+z1b*\
        W1/rb/W7**2*(y1/rb-sinB))))/np.pi/(1-nu))+\
        b3/2*(1/4*((2-2*nu)*rFib_ry1-(2-2*nu)*y2*sinB/W7**2*W2*(y1/rb-\
        sinB)-(2-2*nu)*y2*sinB/W7*a/rb**3*y1-y2*W8*sinB/rb**3/W7*(1+\
        W1/W7*W2+a*y3b/rb2)*y1-y2*W8*sinB/rb/W7**2*(1+W1/W7*W2+\
        a*y3b/rb2)*(y1/rb-sinB)+y2*W8*sinB/rb/W7*(1/rb*cosB*y1/\
        W7*W2-W1/W7**2*W2*(y1/rb-sinB)-W1/W7*a/rb**3*y1-2*a*y3b/\
        rb2**2*y1))/np.pi/(1-nu))

    v23 = b1/2*(1/4*(N1*(((2-2*nu)*cotB**2-nu)*(y3b/rb+1)/W6-((2-2*nu)*\
        cotB**2+1-2*nu)*cosB*W3/W7)+N1/W6**2*(y1*cotB*(1-W5)+nu*y3b-a+\
        y2**2/W6*W4)*(y3b/rb+1)-N1/W6*(1/2*a*y1*cotB/rb**3*2*y3b+\
        nu-y2**2/W6**2*W4*(y3b/rb+1)-1/2*y2**2/W6*a/rb**3*2*y3b)-N1*\
        sinB*cotB/W7*W2+N1*z1b*cotB/W7**2*W2*W3+1/2*N1*z1b*cotB/W7*\
        a/rb**3*2*y3b-a/rb**3*y1*cotB+3/2*a*y1*W8*cotB/rb**5*2*y3b+\
        1/W6*(-2*nu+1/rb*(N1*y1*cotB-a)+y2**2/rb/W6*W5+a*y2**2/\
        rb**3)-W8/W6**2*(-2*nu+1/rb*(N1*y1*cotB-a)+y2**2/rb/W6*W5+\
        a*y2**2/rb**3)*(y3b/rb+1)+W8/W6*(-1/2/rb**3*(N1*y1*cotB-a)*\
        2*y3b-1/2*y2**2/rb**3/W6*W5*2*y3b-y2**2/rb/W6**2*W5*(y3b/\
        rb+1)-1/2*y2**2/rb2**2/W6*a*2*y3b-3/2*a*y2**2/rb**5*2*y3b)+\
        1/W7*(cosB**2-1/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**\
        3-1/rb/W7*(y2**2*cosB**2-a*z1b*cotB/rb*W1))-W8/W7**2*(cosB**2-\
        1/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**3-1/rb/W7*\
        (y2**2*cosB**2-a*z1b*cotB/rb*W1))*W3+W8/W7*(1/2/rb**3*(N1*\
        z1b*cotB+a*cosB)*2*y3b-1/rb*N1*sinB*cotB+a*z1b*cotB/rb**3+a*\
        y3b*sinB*cotB/rb**3-3/2*a*y3b*z1b*cotB/rb**5*2*y3b+1/2/rb**\
        3/W7*(y2**2*cosB**2-a*z1b*cotB/rb*W1)*2*y3b+1/rb/W7**2*(y2**\
        2*cosB**2-a*z1b*cotB/rb*W1)*W3-1/rb/W7*(-a*sinB*cotB/rb*W1+\
        1/2*a*z1b*cotB/rb**3*W1*2*y3b-a*z1b*cotB/rb*(cosB*y3b/rb+\
        1))))/np.pi/(1-nu))+\
        b2/2*(1/4*((2-2*nu)*N1*rFib_ry3*cotB**2-N1*y2/W6**2*((W5-1)*cotB+\
        y1/W6*W4)*(y3b/rb+1)+N1*y2/W6*(-1/2*a/rb**3*2*y3b*cotB-y1/\
        W6**2*W4*(y3b/rb+1)-1/2*y1/W6*a/rb**3*2*y3b)+N1*y2*cotB/\
        W7**2*W9*W3+1/2*N1*y2*cotB/W7*a/rb**3/cosB*2*y3b-a/rb**3*\
        y2*cotB+3/2*a*y2*W8*cotB/rb**5*2*y3b+y2/rb/W6*(N1*cotB-2*\
        nu*y1/W6-a*y1/rb*(1/rb+1/W6))-1/2*y2*W8/rb**3/W6*(N1*\
        cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))*2*y3b-y2*W8/rb/W6**\
        2*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))*(y3b/rb+1)+y2*\
        W8/rb/W6*(2*nu*y1/W6**2*(y3b/rb+1)+1/2*a*y1/rb**3*(1/rb+\
        1/W6)*2*y3b-a*y1/rb*(-1/2/rb**3*2*y3b-1/W6**2*(y3b/rb+\
        1)))+y2*cotB/rb/W7*((-2+2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)-\
        1/2*y2*W8*cotB/rb**3/W7*((-2+2*nu)*cosB+W1/W7*W9+a*y3b/\
        rb2/cosB)*2*y3b-y2*W8*cotB/rb/W7**2*((-2+2*nu)*cosB+W1/W7*\
        W9+a*y3b/rb2/cosB)*W3+y2*W8*cotB/rb/W7*((cosB*y3b/rb+1)/\
        W7*W9-W1/W7**2*W9*W3-1/2*W1/W7*a/rb**3/cosB*2*y3b+a/rb2/\
        cosB-a*y3b/rb2**2/cosB*2*y3b))/np.pi/(1-nu))+\
        b3/2*(1/4*(N1*(-sinB*W3/W7+y1/W6**2*(1+a/rb)*(y3b/rb+1)+\
        1/2*y1/W6*a/rb**3*2*y3b+sinB/W7*W2-z1b/W7**2*W2*W3-1/2*\
        z1b/W7*a/rb**3*2*y3b)+y1/rb*(a/rb2+1/W6)-1/2*y1*W8/rb**\
        3*(a/rb2+1/W6)*2*y3b+y1*W8/rb*(-a/rb2**2*2*y3b-1/W6**2*\
        (y3b/rb+1))-1/W7*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1/\
        rb/W7*(y2**2*cosB*sinB-a*z1b/rb*W1))+W8/W7**2*(sinB*(cosB-\
        a/rb)+z1b/rb*(1+a*y3b/rb2)-1/rb/W7*(y2**2*cosB*sinB-a*z1b/\
        rb*W1))*W3-W8/W7*(1/2*sinB*a/rb**3*2*y3b+sinB/rb*(1+a*y3b/\
        rb2)-1/2*z1b/rb**3*(1+a*y3b/rb2)*2*y3b+z1b/rb*(a/rb2-a*\
        y3b/rb2**2*2*y3b)+1/2/rb**3/W7*(y2**2*cosB*sinB-a*z1b/rb*\
        W1)*2*y3b+1/rb/W7**2*(y2**2*cosB*sinB-a*z1b/rb*W1)*W3-1/\
        rb/W7*(-a*sinB/rb*W1+1/2*a*z1b/rb**3*W1*2*y3b-a*z1b/rb*\
        (cosB*y3b/rb+1))))/np.pi/(1-nu))+\
        b1/2*(1/4*((2-2*nu)*(N1*rFib_ry2*cotB+1/W6*W5-y2**2/W6**2*W5/\
        rb-y2**2/W6*a/rb**3-cosB/W7*W2+y2**2*cosB/W7**2*W2/rb+y2**2*\
        cosB/W7*a/rb**3)+W8/rb*(2*nu/W6+a/rb2)-y2**2*W8/rb**3*(2*\
        nu/W6+a/rb2)+y2*W8/rb*(-2*nu/W6**2/rb*y2-2*a/rb2**2*y2)+\
        W8*cosB/rb/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-y2**2*W8*cosB/\
        rb**3/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-y2**2*W8*cosB/rb2/W7**\
        2*(1-2*nu-W1/W7*W2-a*y3b/rb2)+y2*W8*cosB/rb/W7*(-1/rb*\
        cosB*y2/W7*W2+W1/W7**2*W2/rb*y2+W1/W7*a/rb**3*y2+2*a*\
        y3b/rb2**2*y2))/np.pi/(1-nu))+\
        b2/2*(1/4*((-2+2*nu)*N1*cotB*(1/rb*y2/W6-cosB/rb*y2/W7)+(2-\
        2*nu)*y1/W6**2*W5/rb*y2+(2-2*nu)*y1/W6*a/rb**3*y2-(2-2*\
        nu)*z1b/W7**2*W2/rb*y2-(2-2*nu)*z1b/W7*a/rb**3*y2-W8/rb**\
        3*(N1*cotB-2*nu*y1/W6-a*y1/rb2)*y2+W8/rb*(2*nu*y1/W6**2/\
        rb*y2+2*a*y1/rb2**2*y2)+W8/W7**2*(cosB*sinB+W1*cotB/rb*((2-\
        2*nu)*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))/\
        rb*y2-W8/W7*(1/rb2*cosB*y2*cotB*((2-2*nu)*cosB-W1/W7)-W1*\
        cotB/rb**3*((2-2*nu)*cosB-W1/W7)*y2+W1*cotB/rb*(-cosB/rb*\
        y2/W7+W1/W7**2/rb*y2)-a/rb**3*(sinB-y3b*z1b/rb2-z1b*W1/\
        rb/W7)*y2+a/rb*(2*y3b*z1b/rb2**2*y2-z1b/rb2*cosB*y2/W7+\
        z1b*W1/rb**3/W7*y2+z1b*W1/rb2/W7**2*y2)))/np.pi/(1-nu))+\
        b3/2*(1/4*((2-2*nu)*rFib_ry2+(2-2*nu)*sinB/W7*W2-(2-2*nu)*y2**2*\
        sinB/W7**2*W2/rb-(2-2*nu)*y2**2*sinB/W7*a/rb**3+W8*sinB/rb/\
        W7*(1+W1/W7*W2+a*y3b/rb2)-y2**2*W8*sinB/rb**3/W7*(1+W1/\
        W7*W2+a*y3b/rb2)-y2**2*W8*sinB/rb2/W7**2*(1+W1/W7*W2+a*\
        y3b/rb2)+y2*W8*sinB/rb/W7*(1/rb*cosB*y2/W7*W2-W1/W7**2*\
        W2/rb*y2-W1/W7*a/rb**3*y2-2*a*y3b/rb2**2*y2))/np.pi/(1-nu))
        
    return np.asarray([v11, v22, v33, v12, v13, v23])
