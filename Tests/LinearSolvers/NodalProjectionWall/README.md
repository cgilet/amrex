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
correction and 0 with it. Two more checks without a solve follow:

- **A cell-centred source:** u = (x (y+y^2), y (x+x^2)) with S = div u, so compRHS with the
  source should give 0. It prints max |rhs| / h^2 at interior nodes, on the walls and at the
  (0,0) corner: O(1) would be second order, O(1/h) first. It is 0 in every mode, so the
  velocity and the source are treated consistently.
- **The Burggraf rhs near the corners:** rhs / h^2 on the 3x3 nodes at the (0,0) and the
  (0,n) lid corner.

## Build and run

```shell
make -j4 AMREX_HOME=/path/to/amrex       # DIM=3 for 3D
mpiexec -n 2 ./main2d.gnu.MPI.ex inputs
```

`n_cell`, `max_grid_size`, `rtol`, `verbose`, `maxorders`, `wall_corr` and `divu_check` can be
set in `inputs` or on the command line. `inputs` uses `wall_corr = 2`; add `wall_corr=0` on
the command line to see the error. It takes a few seconds. Apart from the solve's residual,
which is at round-off, the output does not depend on `max_grid_size` (checked with 16, 32 and
1024), and 3D (8 cells in z) prints the same numbers as 2D.

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

In incflo's Burggraf cavity, the gp error at the lid corners is about 0.25 without the fix and
0.165, 0.143 and 0.121 at n = 32, 64 and 128 with corners left alone. An earlier corner
extrapolation made those converge at first order (0.044, 0.023, 0.011), but in a
manufactured incflo test with div(u) = S != 0 the corner errors then stopped converging.
Whether `wall_corr=2` handles both cases is still to be checked in incflo. This test leaves
the corners out of its wall fits.
