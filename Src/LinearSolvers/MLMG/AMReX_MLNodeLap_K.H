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

inline void
mlndlap_scale_neumann_bc (Real s, Box const& bx, Array4<Real> const& rhs, Box const& nddom,
                          GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                          GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc) noexcept
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (lobc[idim] == LinOpBCType::Neumann || lobc[idim] == LinOpBCType::inflow) {
            Box const& blo = amrex::bdryLo(bx, idim);
            if (blo.smallEnd(idim) == nddom.smallEnd(idim)) {
                AMREX_HOST_DEVICE_PARALLEL_FOR_3D (blo, i, j, k,
                {
                    rhs(i,j,k) *= s;
                });
            }
        }
        if (hibc[idim] == LinOpBCType::Neumann || hibc[idim] == LinOpBCType::inflow) {
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

// If the node is on exactly one non-periodic face of the domain and that face is a Neumann
// boundary (a wall), return the direction normal to it and set side to +1 for a low face and -1
// for a high face.  Otherwise return -1.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mlndlap_single_wall_dir (IntVect const& node, Box const& nddom,
                             GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                             GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc,
                             int& side) noexcept
{
    int nbnd = 0;
    int dir = -1;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        bool const lo = node[idim] == nddom.smallEnd(idim) && lobc[idim] != LinOpBCType::Periodic;
        bool const hi = node[idim] == nddom.bigEnd(idim)   && hibc[idim] != LinOpBCType::Periodic;
        if (lo || hi) { ++nbnd; }
        if (lo && lobc[idim] == LinOpBCType::Neumann) { dir = idim; side =  1; }
        if (hi && hibc[idim] == LinOpBCType::Neumann) { dir = idim; side = -1; }
    }
    return (nbnd == 1) ? dir : -1;
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
