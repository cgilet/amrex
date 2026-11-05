# The nodal divergence at walls and inflow faces

Why `MLNodeLaplacian`'s divergence is first order at domain boundaries, why that gives an
O(1) pressure-gradient error next to no-slip walls in viscous flows, how
`MLNodeLaplacian::setWallDivuCorrection` fixes it, and how it treats inflow faces, corners and
outflows. The numbers come from `README.md` in this directory and from incflo; the last section
lists what is still open.

Notation: h is the cell size, u_n the velocity component normal to a boundary, u_t a
tangential one, and u_n'' the second derivative of u_n in the normal direction at the
boundary. Cells (i,j) have centres ((i+1/2)h, (j+1/2)h); node (i,j) is at (ih, jh).

## 1. The symptom

In incflo, in every viscous run with no-slip walls, the wall-normal pressure gradient in the
cells next to a wall has an error that does not converge:

    gp_n - dp/dn = -c mu u_n''

with c = 2/3 for the default tensor viscous solve, 4/9 with `mg_maxorder = 3`, 1/2 for the
scalar solve and 1/3 for the scalar solve with `mg_maxorder = 3` (Burggraf cavity). The
pressure is first order next to walls; the velocity stays second order.

Two errors add up in c. One is the viscous operator's own wall stencil with `maxorder = 2`,
O(1) in the wall cell, which `maxorder = 3` removes. The other is the nodal projection's, and
it is the subject here.

## 2. The nodal divergence is a control-volume integral

`mlndlap_divu` computes, at each node, the divergence theorem over the node's control volume
(the box of side h centred on the node):

    rhs = (1/|V|) (sum over the faces of V of the flux u.n)

Each face of V crosses two cells (in 2D), and the stencil takes the flux through each half of
the face from the centre of the cell it lies in. For the face at x = x_i + h/2 that is

    (h/2) (u(i,j-1) + u(i,j))

the trapezoid rule over [y_j - h/2, y_j + h/2] with the cell centres at its ends. At an
interior node this is second order.

## 3. Wall nodes: where first order comes from

At a node on the wall y = 0, the control volume is cut in half, 0 < y < h/2, and
`mlndlap_impose_neumann_bc` doubles the result to divide by the half volume.

- The wall face, y = 0: the flux is the normal velocity on the wall, 0. Exact.
- The face y = h/2 lies on the centres of the first row of cells. Second order.
- The side faces, x = x_i +- h/2 with 0 < y < h/2, each lie in one cell. The stencil takes the
  flux at that cell's centre, y = h/2, but the segment's midpoint is y = h/4. The difference
  is (h/4) d(u_t)/dn on each side, and across the node it gives a first-order error in the
  divergence:

      rhs_error = (h/4) d/dn (d u_t / dt)

What the projection does with it: it makes the discrete divergence zero, so it shifts the
wall-normal velocity in the first few cells by

    delta = (h^2/8) u_n''

(over the half-cell height h/2, the divergence constraint turns the (h/4) u_n'' error into
this offset). That is second order, but it is nearly the same in each of the first few cells:
it is a step from 0 at the wall to delta inside. A viscous operator divides by h^2, so in the
wall cell it sees an O(1) force, -c' u_n'' with c' = 1/4 (`MLABecLaplacian`, maxorder 2), 1/3
(maxorder 3) or 2/5 (maxorder 4), and 4/3 of these for `MLTensorOp`.

The projection fixes the wall-normal velocity, so the time step cannot absorb an O(1)
wall-normal force with the velocity. The pressure at the wall nodes takes it instead, which
gives the O(1) error in gp_n.

## 4. The correction

Take the side-face fluxes at y = h/4 instead, extrapolated linearly from the first two rows of
cells:

    u_t(h/4) = (5/4) u_t(h/2) - (1/4) u_t(3h/2)      (error (5/32) h^2 u_t'')

For the divergence that adds, after the Neumann doubling, 1/4 [D(row 1) - D(row 2)], where D
is the stencil's tangential derivative across the node in that row. A cell-centred source S
(div u = S) is integrated over the same half volume, from the first row only, and gets the
same treatment.

With the correction, the wall nodes are second order, delta goes to 0 at O(h), and so does c
with `maxorder = 3` (README tables). In incflo's Burggraf cavity with the tensor solve and
`mg_maxorder = 3`, c on the walls is 0.45 to 0.50 without the correction (n = 32, 64), and with
it 0.13, 0.07 on the x walls and below 0.06 on the bottom and the lid.

Implementation (`AMReX_MLNodeLap_2D_K.H`, `_3D_K.H`, `AMReX_MLNodeLaplacian_misc.cpp`): the
correction is added in a separate pass over the wall and inflow nodes only
(`mlndlap_for_each_wall_node`), after the Neumann scaling of `mlndlap_impose_neumann_bc`, so
it does not depend on the scaling factor.

## 5. Corners and edges

At a node on two walls (a corner in 2D, an edge in 3D), the control volume is half a cell
thick in each wall direction, and every term is one-sided in every wall direction it is
tangential to. Mode 2 extrapolates each velocity term of component d along each wall t != d:

- if d is not a wall direction, the tangential derivative across the node;
- if d is a wall direction too, the flux through wall d and the face h/2 from it: the first
  cell minus the ghost cell, which holds the normal velocity on the face.

The source is extrapolated in each wall direction in the same way. This formulation is Marc
Day's (drummerdoc/amrex@36ed98865); the branch implements it in the dimension-specific
kernels above. Mode 1 leaves edge and corner nodes alone.

In the AMReX reproducer, mode 2 brings the projected velocity error in the Burggraf lid corner
cells from 3.8 h^2 (no correction) and 1.6 h^2 (mode 1) to 0.08 h^2 at n = 256. In incflo's
Burggraf cavity, the lid-corner gp error is 0.198, 0.199 (n = 32, 64) without the correction,
0.165, 0.143 with mode 1, and 0.044, 0.023 with mode 2.

## 6. Inflow faces

`mlndlap_divu` treats a face mapped to `LinOpBCType::inflow` like a wall: the control volume
is cut in half, the tangential velocity in the ghost cells is not used (and `compRHS` zeroes
it), and the Neumann scaling doubles the result. incflo maps `mi`, `direction_dependent` and
`mixed` to it. The difference from a wall is the flux through the face itself.

At a node on the inflow face x = 0 (away from the walls):

- The inflow face, x = 0, from y_j - h/2 to y_j + h/2: the ghost cells at y_j +- h/2 hold the
  inflow velocity on the face (the `ext_dir` convention), and the stencil takes the trapezoid
  rule between them. Second order; nothing to fix. This relies on the ghost cells holding
  face values, not extrapolated cell-centre values.
- The face x = h/2 lies on the first column's cell centres. Second order.
- The faces y = y_j +- h/2, with 0 < x < h/2: one-sided, evaluated at x = h/2 instead of h/4.
  The same first-order error as at a wall, and the same correction.

So an inflow face needs exactly the wall correction, and since the correction's flux through a
wall is the first cell minus the ghost cell, it already includes a nonzero inflow velocity.
`setWallDivuCorrection` counts inflow faces as walls.

The tangential velocity on an inflow face (or a no-slip wall) is known in principle, and
interpolating between it and the first cell would also be second order, with a smaller error
constant (h^2/32). It is not used because the data are not reliably there: `compRHS` zeroes
the tangential ghost velocity at inflow faces, incflo resets the velocity ghost cells to 0
before the projection (losing a moving lid's velocity), slip walls have no tangential face
value, and `direction_dependent` faces hold extrapolated values where the flow leaves.
Extrapolating from the interior needs only the normal velocity on the face, which
`mlndlap_divu` already assumes.

In the reproducer's channel (no-slip walls, inflow at x = 0, u_n'' != 0 on the inflow face),
the normal velocity error next to the inflow face, divided by delta, is 0.99 without the
correction and -0.007 with it at n = 256. incflo's viscous solve is Dirichlet at a mass
inflow, as at a wall, so without the correction an inflow with u_n'' != 0 should show the same
O(1) gp_n error as a wall (not yet checked in incflo).

## 7. Where a no-slip wall meets an inflow

The control volume of the corner node is a quarter cell. Its four faces:

- the inflow face, 0 < y < h/2: flux from the inflow ghost cell at y = h/2;
- the wall face, 0 < x < h/2: flux from the wall ghost cell, 0;
- the two faces inside the domain, from the first cell.

The ghost cell outside both faces is never used (`mlndlap_divu` zeroes its contributions, and
the correction does not read it). So when the data do not match, as with a uniform inflow
u = U up to a no-slip wall, nothing at the node has to decide between U and 0: each face takes
its own boundary condition. The inflow flux U h/2 through the half face is the integral of the
prescribed data, and it is the flux incflo's MAC projection and advection see. With mode 2,
the inflow flux at the corner is extrapolated along the wall like the corner's other terms,
and a uniform U stays U, since (5/4) U - (1/4) U = U.

With data that do not match, the exact solution is singular at the corner, and no stencil
converges pointwise in the corner cells. What matters there is the next point.

## 8. Conservation

The correction amounts to redefining the flux through each half-cell face next to a boundary:
evaluated a quarter cell in, by extrapolation. Each such face is shared by two neighbouring
boundary nodes, and both use the same extrapolated value, so the fluxes still telescope: the
sum of rhs over the control volumes equals the flux through the domain boundary, exactly, for
any velocity. That holds where every boundary node is corrected.

It fails where a corrected node borders an uncorrected one. With mode 1, the node next to a
corner is corrected and the corner node is not; their shared half face gets two different
fluxes, which differ by

    (h/2) (E - u_1),  E = (5/4) u_1 - (1/4) u_2,  i.e.  (h/8) (u_1 - u_2)

with u_1, u_2 the first two cells from the other wall. That acts as a mass source at the
corner.

- For a smooth velocity, u_1 - u_2 = O(h) and the source is O(h^2): harmless.
- Where the velocity changes by O(1) from cell to cell next to the corner (a uniform inflow
  against a no-slip wall; a lid-driven cavity with a uniform lid), it is O(h U): first order.

The reproducer measures both:

- Conservation check (random cell velocity, no solve): the sum of rhs over the control volumes
  minus the boundary flux is 1e-16 without the correction and with mode 2, and 1e-2 to 5e-4
  (n = 32 to 256) with mode 1.
- A uniform inflow against no-slip walls, projected: the flux of the projected velocity
  through cross-sections of the channel minus the inflow flux. Each stencil conserves its own
  flux, so it has to be measured with the same face fluxes the stencil uses: h sum_j u(i,j)
  for the original stencil, and with the corrected half-cell faces next to the walls,
  h sum_j u(i,j) + (h/2)(u_1 - u_2)/4 at each wall, for the corrected one (these terms are those
  of the velocity before the projection, since the correction is applied to the right-hand
  side). At x = 1/2:

  | n | no correction | mode 1, own flux | mode 2, own flux | mode 2, h sum_j u |
  |---|---|---|---|---|
  | 32 | -2.8e-12 | -2.3e-3 | -2.8e-12 | 2.0e-4 |
  | 64 | -2.7e-12 | -1.1e-3 | -2.7e-12 | 5.3e-5 |
  | 128 | -2.7e-12 | -5.8e-4 | -2.7e-12 | 1.3e-5 |
  | 256 | -2.3e-12 | -2.9e-4 | -2.3e-12 | 3.4e-6 |

  Mode 2 carries its own flux exactly, like the original stencil. Measured with the original
  stencil's flux instead, it is off by the O(h^2) wall terms, which vary with x: two
  second-order approximations of the same flux differ by that much, but no mass is lost.
  Mode 1 is off at first order, by the same amount at every x in either measure: mass lost at
  the corners.

Before inflow faces were counted as walls, mode 2 had the same first-order loss at wall/inflow
corners, since the corner node was not corrected and its neighbour on the wall was.

So only mode 2 is conservative. Mode 1 is kept as an option and documented as not
conservative at corners. It could be made conservative by not correcting the half faces that
a node shares with a skipped corner node, at the cost of first-order fluxes there.

## 9. Outflows

A pressure outflow is Dirichlet for phi, so its nodes are not solved for and never corrected,
in any mode. The wall nodes next to it are on the wall only and are corrected like any other;
their stencil does not reach past the outflow. In the reproducer's channel, the wall cells
next to the outflow go to 0 at O(h) like the rest of the wall.

The corrected projection leaves a larger but smooth O(h^2) velocity error near the outflow
than the uncorrected one. A smooth O(h^2) error stays O(h^2) under a viscous operator, and
incflo's viscous solve is Neumann at a pressure outflow, so an offset there is not amplified.

## 10. What is still open

- **div(u) = S at corners.** An earlier corner treatment made incflo's corner errors stop
  converging in a manufactured solution with div(u) = S (incflo probtype 605, branch
  `divu_source_validation`). Mode 2 is conservative and treats the velocity and the source alike; whether
  it converges there is the deciding incflo test.
- **3D corners** where three walls meet: the reproducer's conservation check now covers them
  (walls on all six faces; mode 2 conservative to round-off, mode 1 not), but no projection
  test does (the reproducer's solves and the 3D manufactured solution are periodic in one
  direction).
- **incflo inflow cases** with viscosity and u_n'' != 0 at the inflow have not been run.
- **EB** is not supported: the correction applies in regular boxes only, and only the build
  has been checked.
- **AMR**: see section 11. The coarse/fine nodes on a wall are not corrected, but that
  stencil has an O(1) error at the corners of the coarse/fine boundary anyway.
- **GPU**: only a CUDA compile has been checked, not a run.
- **The viscous operator.** With incflo's default `mg_maxorder = 2`, its own O(1) wall error
  remains; `mg_maxorder = 3` is needed for c to go to 0.

## 11. AMR: a fine level on a wall

The correction is added level by level, so the wall nodes of a fine level are corrected like
those of the coarse one, up to the coarse/fine boundary. At a node on the coarse/fine boundary,
`compRHS` then rebuilds the coarse rhs from the uncovered coarse cells and a restriction of the
fine level (`mlndlap_divu_cf_contrib`, `mlndlap_divu_fine_contrib`), which have no correction.
Where that boundary meets a wall, the node is first order.

That turns out not to matter: those nodes are corners of the coarse/fine boundary, and there
the stencil is not even exact for a linear shear. The reproducer, with a fine level (ratio 2)
on the bottom wall, gives for u = (y, 0) (u_n'' = 0, so no wall error) an rhs of -1/8 and
1/8 at the two coarse/fine nodes on the wall, and 1/16 at the fine level's corner away from
the wall, at every resolution and in every mode. By hand at that corner: the three coarse
cells give 1/4 and the restricted fine level -3/16. The stencil takes the velocity in each
cell as constant. For a shear, the error that leaves in each cell's share of the node, of
order h^2 out of a control volume of order h^2, cancels between the four cells of a uniform
node, not where they differ in size. The missing wall correction is O(h) on top of this O(1).

Along the coarse/fine boundary away from its corners, the rhs is O(h), first order, like an
uncorrected wall. Extending the correction to the coarse/fine nodes on a wall would not make
them second order. That would need a coarse/fine stencil that is consistent for linear fields.

## Reproducing

- AMReX: branch `nodal_wall_divu_combined` (not pushed yet), `Tests/LinearSolvers/NodalProjectionWall`:

      make -j4 DIM=2 USE_MPI=FALSE
      ./main2d.gnu.ex inputs wall_corr=0     # 1, 2

- AMReX-Hydro: branch `nodal_wall_divu_correction` (not pushed yet), input
  `nodal_proj.wall_divu_correction` (0, 1, 2).
- incflo: build against both, then add `nodal_proj.wall_divu_correction=2` to a run. The
  Burggraf reproducer is `test_no_eb_2d/wall_gp_error/run.sh` on the `wall_gp_error` branch of
  cgilet/incflo; the manufactured div(u) = S solution is `test_no_eb_2d/inputs.divu_source_mms`
  (probtype 605) on the `divu_source_validation` branch.
