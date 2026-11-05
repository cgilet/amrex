#ifndef AMREX_MLNODELAP_K_H_
#define AMREX_MLNODELAP_K_H_
#include <AMReX_Config.H>

#include <AMReX_FArrayBox.H>
#include <AMReX_LO_BCTYPES.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBCellFlag.H>
#endif
#include <AMReX_MLNodeLinOp_K.H>

namespace amrex {

/// \cond DOXYGEN_IGNORE
namespace nodelap_detail {

#if (BL_USE_FLOAT)
    constexpr float eps = 1.e-30_rt;
#else
    constexpr double eps = 1.e-100_rt;
#endif
    constexpr Real almostone  = Real(1.) - Real(100.)*std::numeric_limits<Real>::epsilon();
    constexpr Real almostzero = Real(1.) - almostone;

}
/// \endcond

// A domain face that mlndlap_divu and mlndlap_scale_neumann_bc treat as a wall: Neumann or inflow.
// The control volumes of its nodes are cut in half.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool mlndlap_wall_bc (LinOpBCType t) noexcept
{
    return t == LinOpBCType::Neumann || t == LinOpBCType::inflow;
}

inline void
mlndlap_scale_neumann_bc (Real s, Box const& bx, Array4<Real> const& rhs, Box const& nddom,
                          GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                          GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc) noexcept
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (mlndlap_wall_bc(lobc[idim])) {
            Box const& blo = amrex::bdryLo(bx, idim);
            if (blo.smallEnd(idim) == nddom.smallEnd(idim)) {
                AMREX_HOST_DEVICE_PARALLEL_FOR_3D (blo, i, j, k,
                {
                    rhs(i,j,k) *= s;
                });
            }
        }
        if (mlndlap_wall_bc(hibc[idim])) {
            Box const& bhi = amrex::bdryHi(bx, idim);
            if (bhi.bigEnd(idim) == nddom.bigEnd(idim)) {
                AMREX_HOST_DEVICE_PARALLEL_FOR_3D (bhi, i, j, k,
                {
                    rhs(i,j,k) *= s;
                });
            }
        }
    }
}

inline void
mlndlap_impose_neumann_bc (Box const& bx, Array4<Real> const& rhs, Box const& nddom,
                           GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                           GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc) noexcept
{
    mlndlap_scale_neumann_bc(2.0, bx, rhs, nddom, lobc, hibc);
}

inline void
mlndlap_unimpose_neumann_bc (Box const& bx, Array4<Real> const& rhs, Box const& nddom,
                           GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                           GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc) noexcept
{
    mlndlap_scale_neumann_bc(0.5, bx, rhs, nddom, lobc, hibc);
}

// The faces at a node that the wall correction of setWallDivuCorrection applies to, walls
// (Neumann) and inflow faces: wall[idim] = +1 for a low face, -1 for a high face, 0 for none.
// mlndlap_divu treats both alike: the control volume is cut in half, and the ghost cells hold
// the normal velocity on the face (zero at a wall).  Returns the number of such faces, or 0 if
// the correction does not apply: the node is on none, it is on another non-periodic face
// (Dirichlet, where the node is not solved for), or it is on two or more (domain edges and
// corners) and mode is 1.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mlndlap_wall_correction_faces (IntVect const& node, Box const& nddom,
                                   GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                                   GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc,
                                   int mode, GpuArray<int,AMREX_SPACEDIM>& wall) noexcept
{
    int nwall = 0;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        wall[idim] = 0;
        if (node[idim] == nddom.smallEnd(idim) && lobc[idim] != LinOpBCType::Periodic) {
            if (!mlndlap_wall_bc(lobc[idim])) { return 0; }
            wall[idim] = 1;
            ++nwall;
        }
        if (node[idim] == nddom.bigEnd(idim) && hibc[idim] != LinOpBCType::Periodic) {
            if (!mlndlap_wall_bc(hibc[idim])) { return 0; }
            wall[idim] = -1;
            ++nwall;
        }
    }
    return (nwall == 1 || (nwall > 1 && mode >= 2)) ? nwall : 0;
}

// Calls f(i,j,k) once at each node of the nodal box bx on a domain face that is a wall
// (mlndlap_wall_bc), i.e. on the faces that mlndlap_scale_neumann_bc scales.  The corrections of
// setWallDivuCorrection are added there, after the scaling, so they do not depend on its factor.
// Only those faces are visited.  A node on several (a domain edge or corner) is visited from its
// lowest wall direction only.
template <typename F>
void mlndlap_for_each_wall_node (Box const& bx, Box const& nddom,
                                 GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                                 GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc, F const& f)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (int side = 0; side < 2; ++side) {
            bool const lo = (side == 0);
            if (!mlndlap_wall_bc(lo ? lobc[idim] : hibc[idim])) { continue; }
            Box const b = lo ? amrex::bdryLo(bx, idim) : amrex::bdryHi(bx, idim);
            if (lo ? (b.smallEnd(idim) != nddom.smallEnd(idim))
                   : (b.bigEnd(idim) != nddom.bigEnd(idim))) { continue; }
            amrex::ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                IntVect const iv(AMREX_D_DECL(i,j,k));
                bool first = true;
                for (int d = 0; d < idim; ++d) {
                    if ((iv[d] == nddom.smallEnd(d) && mlndlap_wall_bc(lobc[d])) ||
                        (iv[d] == nddom.bigEnd(d) && mlndlap_wall_bc(hibc[d]))) {
                        first = false;
                    }
                }
                if (first) { f(i,j,k); }
            });
        }
    }
}

}

#if (AMREX_SPACEDIM == 1)
#include <AMReX_MLNodeLap_1D_K.H>
#elif (AMREX_SPACEDIM == 2)
#include <AMReX_MLNodeLap_2D_K.H>
#else
#include <AMReX_MLNodeLap_3D_K.H>
#endif

namespace amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlndlap_normalize_sten (int i, int j, int k, Array4<Real> const& x,
                             Array4<Real const> const& sten,
                             Array4<int const> const& msk, Real s0_norm0) noexcept
{
    if (!msk(i,j,k) && std::abs(sten(i,j,k,0)) > s0_norm0) {
        x(i,j,k) /= sten(i,j,k,0);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlndlap_jacobi_sten (int i, int j, int k, Array4<Real> const& sol,
                          Real Ax, Array4<Real const> const& rhs,
                          Array4<Real const> const& sten,
                          Array4<int const> const& msk) noexcept
{
    if (msk(i,j,k)) {
        sol(i,j,k) = Real(0.0);
    } else if (sten(i,j,k,0) != Real(0.0)) {
        sol(i,j,k) += Real(2./3.) * (rhs(i,j,k) - Ax) / sten(i,j,k,0);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlndlap_jacobi_sten (Box const& bx, Array4<Real> const& sol,
                          Array4<Real const> const& Ax,
                          Array4<Real const> const& rhs,
                          Array4<Real const> const& sten,
                          Array4<int const> const& msk) noexcept
{
    amrex::LoopConcurrent(bx, [=] (int i, int j, int k) noexcept
    {
        if (msk(i,j,k)) {
            sol(i,j,k) = Real(0.0);
        } else if (sten(i,j,k,0) != Real(0.0)) {
            sol(i,j,k) += Real(2./3.) * (rhs(i,j,k) - Ax(i,j,k)) / sten(i,j,k,0);
        }
    });
}

AMREX_FORCE_INLINE
bool mlndlap_any_fine_sync_cells (Box const& bx, Array4<int const> const& msk, int fine_flag) noexcept
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    for (int k = lo.z; k <= hi.z; ++k) {
    for (int j = lo.y; j <= hi.y; ++j) {
    for (int i = lo.x; i <= hi.x; ++i) {
        if (msk(i,j,k) == fine_flag) { return true; }
    }}}
    return false;
}

}

#endif
