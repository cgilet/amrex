# Nodal projection next to a no-slip wall: O(1) error in the viscous term

This test reproduces, with AMReX alone, an error that incflo shows in every viscous run with
no-slip walls: in the cells next to the wall, the wall-normal pressure gradient has an O(1)
error that does not converge. The incflo side, which shows the error with the unchanged
Burggraf benchmark, is in `test_no_eb_2d/wall_gp_error/` on the `wall_gp_error` branch of
https://github.com/cgilet/incflo.

## What it does

`main.cpp` takes the Burggraf velocity on the unit square, which is divergence-free and zero
on the walls except for the lid's tangential velocity. In 3D the velocity does not depend on
z, w = 0, and z is periodic. For n = 32, 64, 128 and 256 cells per side it:

1. projects the velocity once with `MLNodeLaplacian`, as AMReX-Hydro's `NodalProjector` does
   in incflo (walls are Neumann for phi, sigma = 1, zero velocity in the ghost cells), with
   the wall correction `MLNodeLaplacian::setWallDivuCorrection(wall_corr)`: 0 off, 1 at nodes
   on one wall, 2 also at corners, and
2. applies the viscous operators incflo uses, with Dirichlet walls and each maxorder in
   `maxorders`, to the exact velocity and to the projected one. The operators are
   `MLABecLaplacian` on each component (incflo's scalar solve) and `MLTensorOp` (incflo's
   default tensor solve).

It prints, in the wall cells and leaving out n/8 cells next to each corner:

- **The projected wall-normal velocity error** in the first four cells from the wall, divided
  by delta = (h^2/8) u_n''. Here u_n'' is the second normal derivative of the normal velocity
  at the wall.
- **The operator's wall error,** as the c in (L_h u - lap u)_n = -c u_n''.

It also prints the projected velocity error in the four corner cells divided by h^2, and the
solve's residual and rhs at the (0,n) lid corner node and its two neighbours.

With `divu_check = 1`, it first checks the nodal divergence stencil directly, with no solve:
it calls `compRHS` for u = (x y, -y^2/2), which is divergence-free with u_n'' = -1 on the bottom
wall. It prints the largest |rhs| at interior nodes, which is 0 because the stencil is exact
for quadratics there, and rhs / (h/4) on the bottom-wall nodes. That is 1 without the wall
correction and 0 with it. Three more checks without a solve follow:

- **A cell-centred source:** u = (x (y+y^2), y (x+x^2)) with S = div u, so compRHS with the
  source should give 0. It prints max |rhs| / h^2 at interior nodes, on the walls and at the
  (0,0) corner: O(1) would be second order, O(1/h) first. It is 0 in every mode, so the
  velocity and the source are treated consistently.
- **Conservation:** with a random cell velocity, which changes by O(1) from cell to cell as it
  does next to a corner where the boundary data do not match, the sum of rhs over the nodes'
  control volumes minus the flux through the domain boundary. It is zero to round-off for any
  velocity if the fluxes through the faces of the control volumes telescope, as they do with
  the original stencil. It is checked in a closed box, and with walls at y = 0 and 1 and
  inflow faces at both x ends. In 3D it is also checked with walls at both z ends instead of
  periodic z, so that it includes nodes on three walls, and on two walls and an inflow face.
- **Two AMR levels:** a fine level (ratio 2) touching the bottom wall over the coarse cells
  [3n/8, 5n/8) x [0, n/4), with u = (x y, -y^2/2) and with the linear shear u = (y, 0). It
  prints rhs / (h/4) on the coarse and the fine wall nodes away from the coarse/fine boundary,
  the rhs at the corners of the coarse/fine boundary (the two on the wall, and the fine
  level's corner away from it), and the largest rhs / (h/4) at the other coarse/fine nodes.
  See "Two AMR levels" below.
- **A closed box in 3D** (`box3d = 1`, resolutions `box3d_n_cell`, 3D only): walls on all six
  faces, so that cells sit on two walls (edges) and three (corners). It projects a
  divergence-free velocity that depends on all three coordinates and applies the viscous
  operators, as for Burggraf. See "Closed box in 3D" below.
- **The Burggraf rhs near the corners:** rhs / h^2 on the 3x3 nodes at the (0,0) and the
  (0,n) lid corner.

## Build and run

```shell
make -j4 AMREX_HOME=/path/to/amrex       # DIM=3 for 3D
mpiexec -n 2 ./main2d.gnu.MPI.ex inputs
```

`n_cell`, `max_grid_size`, `rtol`, `verbose`, `maxorders`, `wall_corr`, `divu_check`,
`channel`, `box3d` and `box3d_n_cell` can be set in `inputs` or on the command line. `inputs` uses `wall_corr = 2`; add `wall_corr=0` on
the command line to see the error. It takes a few seconds. Apart from the numbers that are
zero to round-off or to the solver tolerance (the solve's residual, and the conservation and
uniform-inflow flux errors where they are zero), the output does not depend on
`max_grid_size` (checked with 16, 32 and 1024), and 3D (8 cells in z) prints the same numbers
as 2D. The exceptions are the conservation check with `wall_corr=1`, whose random velocity
depends on z, and its two extra 3D columns with walls at both z ends.

## Output without the wall correction (`wall_corr=0`, excerpt)

```
Nodal divergence of u = (x y, -y^2/2) (div u = 0, u_n'' = -1 on the bottom
wall): max |rhs| at interior nodes, and rhs / (h/4) on the bottom-wall
nodes away from the corners (mean, min, max)
     n   max interior         mean          min          max
    32      0.000e+00       1.0000       1.0000       1.0000
    64      0.000e+00       1.0000       1.0000       1.0000
   128      0.000e+00       1.0000       1.0000       1.0000
   256      0.000e+00       1.0000       1.0000       1.0000


Projected Burggraf velocity: max error, and the wall-normal error in the
cells 0-3 from the wall divided by delta = (h^2/8) u_n''
     n   max |u_P - u|   cell 0   cell 1   cell 2   cell 3   corner cells / h^2: (0,0) (n,0) (0,n) (n,n)
    32       2.855e-03    0.903    0.735    0.599    0.489     0.1749   0.1749   2.9234   2.9234
    64       8.132e-04    0.950    0.858    0.774    0.699     0.1138   0.1138   3.3309   3.3309
   128       2.196e-04    0.975    0.926    0.880    0.836     0.0703   0.0703   3.5984   3.5984
   256       5.745e-05    0.987    0.962    0.938    0.914     0.0420   0.0420   3.7652   3.7652

Viscous operator in the wall cells: (L_h u - lap u)_n = -c u_n''
  operator                         n   c, exact u   c, projected u
  MLABecLaplacian, maxorder 2     32       0.2495           0.4987
                                  64       0.2499           0.4997
                                 128       0.2500           0.4999
                                 256       0.2500           0.5000
  MLABecLaplacian, maxorder 3     32       0.0190           0.3505
                                  64       0.0097           0.3426
                                 128       0.0049           0.3381
                                 256       0.0025           0.3358
  MLABecLaplacian, maxorder 4     32       0.0002           0.3983
                                  64       0.0001           0.3996
                                 128       0.0000           0.3999
                                 256       0.0000           0.4000
  MLTensorOp,      maxorder 2     32       0.3269           0.6573
                                  64       0.3305           0.6626
                                 128       0.3320           0.6648
                                 256       0.3327           0.6658
  MLTensorOp,      maxorder 3     32       0.0340           0.4724
                                  64       0.0176           0.4596
                                 128       0.0090           0.4523
                                 256       0.0045           0.4485
  MLTensorOp,      maxorder 4     32       0.0094           0.5351
                                  64       0.0048           0.5350
                                 128       0.0025           0.5344
                                 256       0.0012           0.5339
```

## What it shows

1. **The projection.** After the projection the wall-normal velocity is too large by
   delta = (h^2/8) u_n'' in the first few cells: the ratio in the table tends to 1. This error
   is second order, but it is nearly the same in each cell, so a 1/h^2 stencil does not see
   it as smooth. In the wall cell, with the Dirichlet ghost cell, the Laplacian of the offset
   is -2 delta/h^2 = -u_n''/4 with maxorder 2, -(8/3) delta/h^2 = -u_n''/3 with maxorder 3,
   and -(16/5) delta/h^2 = -2u_n''/5 with maxorder 4. `MLTensorOp` gives 4/3 of these in the
   wall-normal component.

   The cause is `mlndlap_divu` (`AMReX_MLNodeLap_2D_K.H`, `AMReX_MLNodeLap_3D_K.H`). A node
   on a wall has a control volume of half a cell, 0 < n < h/2. The stencil takes the
   tangential derivative of the tangential velocity at the first cell centres, n = h/2, for
   the whole half volume, instead of its average over it (at n = h/4 to second order). The
   two differ by (h/4) u_n''. Over the half-cell height h/2, the divergence constraint turns
   this into the offset delta = (h^2/8) u_n''.

2. **The viscous operator.** With maxorder 2 (incflo's default), the Dirichlet stencil is
   O(1) wrong in the wall cell even for the exact velocity: c = 1/4 for `MLABecLaplacian` and
   1/3 for `MLTensorOp`. With maxorder 3 or 4 this goes away (O(h)).

In the cells next to the wall, the two errors add up:

| operator | maxorder | c, exact u | c, projected u |
|---|---|---|---|
| `MLABecLaplacian` | 2 | 1/4 | 1/2 |
| `MLABecLaplacian` | 3 | 0 | 1/3 |
| `MLABecLaplacian` | 4 | 0 | 2/5 |
| `MLTensorOp` | 2 | 1/3 | 2/3 |
| `MLTensorOp` | 3 | 0 | 4/9 |
| `MLTensorOp` | 4 | 0 | 8/15 |

## How it shows up in incflo

The projection fixes the wall-normal velocity, so the time step cannot correct an O(1) error
in the wall-normal viscous force with the velocity. The wall-node pressure takes it instead,
which makes the pressure first order there and the pressure gradient O(1) wrong in the wall
cells:

gp_n - dp/dn = -c mu u_n''

with the c of the last column above. In incflo's Burggraf cavity at n = 128 on stock AMReX,
the c fitted on the lid is 0.500, 0.336, 0.666 and 0.450 for maxorder 2 and 3 with
`MLABecLaplacian` and `MLTensorOp`, in that order.
Velocity stays second order.

## The fix, and what is still open

`MLNodeLaplacian::setWallDivuCorrection(true)` makes the nodal divergence second order at
walls. At nodes on exactly one non-periodic domain face, `MLNodeLaplacian::compRHS` adds a
correction that takes the tangential velocity (and the cell-centred source, if any) a quarter
cell from the wall, extrapolated linearly from the first two rows of cells. With it and
maxorder 3 or 4, c goes to 0 at O(h) along the walls (`wall_corr=1`, excerpt; `wall_corr=2`
gives nearly the same along the walls):

```
Projected Burggraf velocity: max error, and the wall-normal error in the
cells 0-3 from the wall divided by delta = (h^2/8) u_n''
     n   max |u_P - u|   cell 0   cell 1   cell 2   cell 3   corner cells / h^2: (0,0) (n,0) (0,n) (n,n)
    32       1.730e-03    0.071   -0.006   -0.068   -0.118     0.0092   0.0092   1.7719   1.7719
    64       4.130e-04    0.036   -0.007   -0.045   -0.080     0.0009   0.0009   1.6916   1.6916
   128       9.948e-05    0.018   -0.005   -0.026   -0.046     0.0016   0.0016   1.6300   1.6300
   256       2.429e-05    0.009   -0.003   -0.014   -0.025     0.0012   0.0012   1.5920   1.5920

Viscous operator in the wall cells: (L_h u - lap u)_n = -c u_n''
  operator                         n   c, exact u   c, projected u
  MLABecLaplacian, maxorder 2     32       0.2495           0.2770
                                  64       0.2499           0.2642
                                 128       0.2500           0.2573
                                 256       0.2500           0.2537
  MLABecLaplacian, maxorder 3     32       0.0190           0.0556
                                  64       0.0097           0.0288
                                 128       0.0049           0.0147
                                 256       0.0025           0.0074
  MLABecLaplacian, maxorder 4     32       0.0002           0.0445
                                  64       0.0001           0.0231
                                 128       0.0000           0.0117
                                 256       0.0000           0.0059
  MLTensorOp,      maxorder 2     32       0.3269           0.3662
                                  64       0.3305           0.3509
                                 128       0.3320           0.3424
                                 256       0.3327           0.3380
  MLTensorOp,      maxorder 3     32       0.0340           0.0882
                                  64       0.0176           0.0458
                                 128       0.0090           0.0234
                                 256       0.0045           0.0118
  MLTensorOp,      maxorder 4     32       0.0094           0.0761
                                  64       0.0048           0.0394
                                 128       0.0025           0.0200
                                 256       0.0012           0.0101
```

With `divu_check`, rhs / (h/4) on the bottom-wall nodes is then 0 instead of 1.

**Corners.** With `wall_corr=1`, the correction keeps the original stencil at corner nodes.
That leaves a large error in the corner cells wherever u_n'' is not zero at the corner, as at
the lid corners (0,n) and (n,n). `wall_corr=2` also applies the extrapolation at corners, in
each wall direction, for the velocity and the source in the same way. The projected velocity
error in the lid corner cells, divided by h^2:

| n | `wall_corr=0` | `wall_corr=1` | `wall_corr=2` |
|---|---|---|---|
| 32 | 2.92 | 1.77 | 0.50 |
| 64 | 3.33 | 1.69 | 0.28 |
| 128 | 3.60 | 1.63 | 0.15 |
| 256 | 3.77 | 1.59 | 0.08 |

The solve still converges at the corner nodes (residual at round-off).

Only `wall_corr=2` keeps the divergence conservative. A node next to a corner shares half-cell
faces with the corner node, and with `wall_corr=1` only the first one corrects the flux
through them. The conservation check shows it, with a random velocity:

```
     n     closed box    with inflow faces
    32      1.386e-02      1.386e-02
    64     -2.502e-03     -2.502e-03
   128      5.211e-04      5.211e-04
   256      5.673e-04      5.673e-04
```

with `wall_corr=1`, against 1e-16 with `wall_corr=0` and `wall_corr=2`. The same holds in 3D
with walls on all six faces, where the corners of the box are on three walls (`wall_corr=1`,
then `wall_corr=2`; also with `max_grid_size=4`, two boxes in z):

```
     n     closed, z walls   inflow, z walls
    32       -6.371e-04        -6.371e-04
    64        1.312e-03         1.312e-03
   128        1.666e-04         1.666e-04
   256        1.321e-04         1.321e-04

    32       -1.843e-17         4.163e-17
    64       -1.205e-16         1.041e-16
   128        1.122e-16         1.214e-16
   256        8.632e-17        -2.949e-17
```
 For a smooth velocity
this is an O(h^2) error at each corner, which the Burggraf results above do not show. Where
the velocity changes by O(1) from cell to cell next to a corner, it is an O(h) mass source
(see the uniform inflow below).

In incflo's Burggraf cavity, the gp error at the lid corners is about 0.25 without the fix and
0.165, 0.143 and 0.121 at n = 32, 64 and 128 with corners left alone. An earlier corner
extrapolation made those converge at first order (0.044, 0.023, 0.011), but in a
manufactured incflo test with div(u) = S != 0 the corner errors then stopped converging.
Whether `wall_corr=2` handles both cases is still to be checked in incflo. This test leaves
the corners out of its wall fits.

## Two AMR levels

The correction is applied level by level. At a node on the coarse/fine boundary, `compRHS`
then rebuilds the coarse rhs from the coarse cells and the fine level
(`mlndlap_divu_cf_contrib`, `mlndlap_divu_fine_contrib`), without it. With a fine level on the
bottom wall (the same output in 2D and 3D, z periodic):

```
               wall / (h/4)              rhs at the c/f corners             other c/f
     n      coarse       fine     wall i=3n/8  wall i=5n/8  (3n/8,n/4)     / (h/4)
  u = (x y, -y^2/2)
    32      0.0000     0.0000     -4.199e-02    8.301e-02    2.490e-02      0.3750
    64      0.0000     0.0000     -4.443e-02    8.057e-02    2.417e-02      0.3750
   128      0.0000     0.0000     -4.565e-02    7.935e-02    2.380e-02      0.3750
   256      0.0000     0.0000     -4.626e-02    7.874e-02    2.362e-02      0.3750
  u = (y, 0)
    32      0.0000     0.0000     -1.250e-01    1.250e-01    6.250e-02      0.0000
   ...
```

with `wall_corr=2` (and 1). With `wall_corr=0` the first two columns are 1 for the first
velocity; nothing else changes in any mode.

- The wall nodes of both levels are corrected, the fine ones up to the coarse/fine boundary.
- The corners of the coarse/fine boundary, on the wall or not, have an O(1) rhs, even for the
  linear shear u = (y, 0), which has u_n'' = 0 and needs no correction. The stencil takes the
  velocity in each cell as constant. For a shear, the error that leaves in each cell's share
  cancels between the four cells around a uniform node, but not where they differ in size.
  The missing wall correction at the two coarse/fine nodes on the wall is O(h), next to this
  O(1).
- Along the coarse/fine boundary away from its corners the rhs is O(h) (0.375 h/4), first
  order, as at an uncorrected wall.

So a fine level on a wall gets the correction wherever it can make a difference; the
coarse/fine nodes on the wall are limited by the coarse/fine stencil itself.

## Closed box in 3D

The periodic-z cases have no cell on more than two walls, and Burggraf in 3D does not depend
on z. The closed box projects

    u = grad(Phi) x (1, 2, 3),   Phi = F(x) F(y) H(z),   F = 16 s^2 (1-s)^2,   H = 8 s^2 (1-s)

on [0,1]^3 with walls on all six faces. It is divergence-free and zero on five of them; at
z = 1 it moves tangentially, proportional to F(x) F(y), like the Burggraf lid. The normal
velocity grows like the square of the distance from each wall, so u_n'' != 0 on every face.
No-slip on two walls forces u_n'' to 0 along the edge between them; the lid keeps it nonzero
along the lid's four edges (at its corners it still goes to 0, quadratically). The exact
Laplacian, grad(lap Phi) x (1,2,3), is known in closed form.

Nodal divergence of the exact velocity, max |rhs| / h (O(1) first order, O(h) second):

```
            interior        face        edge      corner
wall_corr=0
    16       11.9458     91.4926    113.3872      2.4797
    32        6.8924    107.0639    120.3903      0.6824
    64        3.7070    116.7839    124.1034      0.1789
wall_corr=1: faces as with 2, edges as with 0
wall_corr=2
    16       11.9458     23.6854     27.5970      3.6511
    32        6.8924     13.5829     15.3610      1.1841
    64        3.7070      7.3503      8.0914      0.3340
   128        1.9244      3.8319      4.1545      0.0885
```

With `wall_corr=2` the divergence is second order on the faces and along the edges. The corner
values are small in every mode, since u_n'' goes to 0 at the corners of this velocity.

Projected velocity, and `MLABecLaplacian` with maxorder 3 (`MLTensorOp` is similar). "cell 0"
is the wall-normal error / delta on the faces, c is fitted to the normal component on the
faces, and the other columns are maxima over the face cells (tangential components) and the
edge cells (all components), exact and projected velocity:

```
     n   cell 0   c, exact   c, proj    face t., exact   proj    edge, exact   proj
wall_corr=0
    16    0.763     0.052     0.376        2.36e+01   8.36e+01     2.90e+01   2.77e+02
    32    0.875     0.028     0.358        1.25e+01   8.17e+01     1.52e+01   3.00e+02
    64    0.935     0.014     0.347        6.46e+00   8.20e+01     7.81e+00   3.16e+02
   128    0.967     0.007     0.340        3.28e+00   8.20e+01     3.95e+00   3.26e+02
wall_corr=2
    16    0.204     0.052     0.150        2.36e+01   3.95e+01     2.90e+01   1.13e+02
    32    0.117     0.028     0.081        1.25e+01   4.12e+01     1.52e+01   6.52e+01
    64    0.063     0.014     0.042        6.46e+00   4.32e+01     7.81e+00   5.56e+01
   128    0.033     0.007     0.021        3.28e+00   4.46e+01     3.95e+00   6.47e+01
```

(n = 128 with `box3d_n_cell="64 128" maxorders=3`; about 1.3 GB.)

- **Faces, normal component:** as in 2D. Without the correction the error is delta and c is
  O(1); with it both go to 0 at O(h).
- **Faces, tangential components:** the viscous error of the projected velocity is O(1) in
  every mode (82 without the correction, 45 with it). The projection leaves an O(h^2)
  tangential velocity at the wall (max |u_P - u| / h^2 in the face cells is about 16 for the
  tangential components with `wall_corr=2`, against 3.7 and falling for the normal one), and
  the Dirichlet viscous stencil, which takes the wall value as 0, turns an O(h^2) value at the
  wall into an O(1) second difference in the wall cell. This is not specific to 3D or to
  the edges: the 2D Burggraf case has it too (with maxorder 3, max over the wall cells away
  from the corners, n = 32 to 256: 3.6 to 4.2 with `wall_corr=2`, 5.4 to 5.3 with 0, for the
  tangential component; the normal one goes from 1.19 to 0.17 with 2). The 2D output above
  fits the normal component only.
- **Edges:** every component is tangential to at least one of the two walls, so the edge
  cells carry the same O(1) error. It is about 5 times smaller with the correction (65
  against 326 at n = 128) and, like the tangential face error, does not go to 0. With
  `wall_corr=1` the edges are not corrected and the edge error is 198 to 234.
- **Corners:** u_n'' goes to 0 at the corners of this velocity, so they test little; the
  projected-velocity viscous error there is 19 to 9 with `wall_corr=0`, 3.4 to 2.1 with 2
  (n = 64, 128).

## Where a wall meets an inflow or an outflow face

With `channel = 1` (in `inputs`), the test also projects a divergence-free channel flow on the
unit square, from the stream function psi = H(y) + F(x) G(y) with H = 3y^2 - 2y^3,
F = x + x^2/2 and G = y^2 (1-y)^2:

- no-slip walls at y = 0 and 1 (Neumann for phi), where v_n'' = -2 (1+x) is not zero up to
  either end,
- an inflow face at x = 0 (`LinOpBCType::inflow`, the inflow velocity in the ghost cells),
  where u_n'' = G'(y),
- a pressure outflow at x = 1 (phi = 0, `LinOpBCType::Dirichlet`),

as incflo sets up mass inflow and pressure outflow. It prints the divergence of the exact
velocity at the nodes, and the normal velocity error after the projection divided by delta,
in the wall cells and next to the inflow face. The outflow face is left out of the second
table: incflo's viscous solve is Neumann there, so an offset of the velocity next to it does
not become an O(1) force. Without the wall correction (`wall_corr=0`):

```
     n   interior       wall   wall i=1   corner i=0   inflow face   wall i=n-1
    32     0.0577     0.8796     0.4838       0.4690        0.0672       0.9236
    64     0.0300     0.9084     0.4920       0.4844        0.0574       0.9614
   128     0.0153     0.9229     0.4961       0.4922        0.0527       0.9806
   256     0.0077     0.9302     0.4980       0.4961        0.0504       0.9903

     n   max |u_P - u|    walls    i=0      i=1      i=2    i=n-2    i=n-1   inflow
    32       5.287e-04    0.929    0.968    0.951    0.945    0.798    0.378    0.892
    64       1.819e-04    0.964    0.990    0.980    0.977    0.808    0.383    0.944
   128       5.843e-05    0.982    0.998    0.993    0.991    0.811    0.385    0.972
   256       1.791e-05    0.991    1.001    0.998    0.997    0.813    0.386    0.986
```

`mlndlap_divu` treats an inflow face like a wall: the control volumes of its nodes are cut in
half in the same way, and the ghost cells hold the normal velocity on the face. The wall
correction does the same, with the flux through the face taken as the first cell minus the
ghost cell. With `wall_corr=1`:

```
     n   interior       wall   wall i=1   corner i=0   inflow face   wall i=n-1
    32     0.0577     0.1119     0.0616       0.4690        0.0236       0.1175
    64     0.0300     0.0573     0.0310       0.4844        0.0118       0.0606
   128     0.0153     0.0290     0.0156       0.4922        0.0059       0.0308
   256     0.0077     0.0146     0.0078       0.4961        0.0029       0.0155

     n   max |u_P - u|    walls    i=0      i=1      i=2    i=n-2    i=n-1   inflow
    32       3.048e-04    0.133    0.428    0.148    0.135    0.135    0.065   -0.074
    64       9.152e-05    0.069    0.407    0.094    0.079    0.071    0.034   -0.033
   128       2.524e-05    0.035    0.397    0.067    0.051    0.037    0.018   -0.015
   256       6.652e-06    0.018    0.392    0.053    0.037    0.019    0.009   -0.007
```

and with `wall_corr=2`:

```
     n   interior       wall   wall i=1   corner i=0   inflow face   wall i=n-1
    32     0.0577     0.1119     0.0616       0.0602        0.0236       0.1175
    64     0.0300     0.0573     0.0310       0.0307        0.0118       0.0606
   128     0.0153     0.0290     0.0156       0.0155        0.0059       0.0308
   256     0.0077     0.0146     0.0078       0.0078        0.0029       0.0155

     n   max |u_P - u|    walls    i=0      i=1      i=2    i=n-2    i=n-1   inflow
    32       3.058e-04    0.132    0.117    0.117    0.117    0.135    0.065   -0.048
    64       9.167e-05    0.069    0.059    0.059    0.059    0.071    0.034   -0.025
   128       2.525e-05    0.035    0.029    0.029    0.029    0.037    0.018   -0.013
   256       6.654e-06    0.018    0.015    0.015    0.015    0.019    0.009   -0.007
```

- **Outflow:** a pressure outflow is Dirichlet for phi, so its nodes are not solved for and
  not corrected. The wall nodes next to it are, and the wall cells next to the outflow go to 0
  at O(h) like the rest of the wall.
- **Inflow face:** second order in both modes, and the normal velocity next to it converges
  to the exact one at O(h) relative to delta. Without the correction it is off by delta, as at
  a wall. incflo's viscous solve is Dirichlet at a mass inflow, as at a wall, so with
  viscosity and u_n'' != 0 at an inflow, that should give the same O(1) error in the normal
  pressure gradient there (not checked in incflo).
- **Wall/inflow corner:** with `wall_corr=1` the corner node keeps the original stencil, and
  the wall cell next to it keeps about 0.4 delta, which does not converge. With `wall_corr=2`
  the corner node is second order too, and the wall cells at the inflow end go to 0 at O(h).

The corner node never uses the ghost cell outside both faces, so it does not matter that the
inflow and the wall disagree there. It takes the inflow flux from the inflow face and the
wall flux from the wall. With `wall_corr=2` the inflow flux at the corner node is
extrapolated along the wall like the rest of the corner's terms, which keeps a uniform inflow
velocity as it is.

### A uniform inflow against no-slip walls

The channel again, with u = 1 in the inflow ghost cells up to the walls, as in a channel with
a uniform `mi` inflow. The data do not match at the wall/inflow corners, and the velocity is
singular there. The velocity projected is u = (2/pi)^2 atan2(y, x) atan2(1-y, x), v = 0,
which is 1 on the inflow face, 0 on the walls, and changes by O(1) from cell to cell next to
the corners at every resolution. The projected velocity must carry the inflow flux, 1,
through every cross-section. The test prints, at x = 1/4, 1/2 and 3/4, minus 1:

- the flux h sum_j u(i,j), which is the flux through a column of control-volume faces as the
  uncorrected stencil defines it, and
- the flux as the corrected stencil defines it. The correction takes the flux through the
  half-cell faces next to a wall as (5/4) u_1 - (1/4) u_2 instead of u_1 (u_1, u_2: the first
  two cells from the wall), which adds (h/2)(u_1 - u_2)/4 at each wall. It is applied to the
  right-hand side, so these terms are those of the velocity before the projection.

A conservative stencil makes its own flux exact, to the solver tolerance:

```
                   h sum_j u                         conserved by the stencil
     n     x = 1/4    x = 1/2    x = 3/4         x = 1/4    x = 1/2    x = 3/4
wall_corr=0
    32  -4.577e-12 -2.822e-12 -1.770e-12      -4.577e-12 -2.822e-12 -1.770e-12
    64  -4.804e-12 -2.713e-12 -1.719e-12      -4.804e-12 -2.713e-12 -1.719e-12
   128  -4.771e-12 -2.667e-12 -1.567e-12      -4.771e-12 -2.667e-12 -1.567e-12
   256  -4.552e-12 -2.300e-12 -1.092e-12      -4.552e-12 -2.300e-12 -1.092e-12
wall_corr=1
    32  -1.805e-03 -2.076e-03 -2.166e-03      -2.281e-03 -2.281e-03 -2.281e-03
    64  -1.022e-03 -1.094e-03 -1.117e-03      -1.147e-03 -1.147e-03 -1.147e-03
   128  -5.430e-04 -5.616e-04 -5.675e-04      -5.750e-04 -5.750e-04 -5.750e-04
   256  -2.798e-04 -2.845e-04 -2.860e-04      -2.879e-04 -2.879e-04 -2.879e-04
wall_corr=2
    32   4.752e-04  2.042e-04  1.143e-04      -4.575e-12 -2.814e-12 -1.763e-12
    64   1.253e-04  5.295e-05  2.959e-05      -4.794e-12 -2.698e-12 -1.708e-12
   128   3.208e-05  1.347e-05  7.523e-06      -4.762e-12 -2.671e-12 -1.578e-12
   256   8.111e-06  3.395e-06  1.896e-06      -4.553e-12 -2.297e-12 -1.104e-12
```

- The original stencil and `wall_corr=2` are conservative: each carries its own flux exactly.
  With `wall_corr=2`, h sum_j u differs from it by the O(h^2) wall terms, which vary with x
  as the velocity gradient at the walls does. That is a difference between two second-order
  approximations of the flux through the cross-section, not lost mass.
- With `wall_corr=1` the wall/inflow corner nodes are not corrected and their neighbours on
  the walls are, and the two use different fluxes through the half-cell faces they share.
  The difference is a mass source at each corner. The flux is off by the same amount at every
  x, in both measures, and that amount converges only at first order.
