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

namespace nodelap_detail {

// A node on exactly one Neumann (wall) domain face: returns the normal direction and sets
// side (0 lo, 1 hi); -1 if the node is on no wall, or on two or more (corners/edges).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int mlndlap_single_wall (IntVect const& iv, Box const& nddom,
                         GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                         GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc, int& side) noexcept
{
    int nwall = 0, nd = -1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (lobc[d] == LinOpBCType::Neumann && iv[d] == nddom.smallEnd(d)) { ++nwall; nd = d; side = 0; }
        if (hibc[d] == LinOpBCType::Neumann && iv[d] == nddom.bigEnd(d))   { ++nwall; nd = d; side = 1; }
    }
    return (nwall == 1) ? nd : -1;
}

}

// Second-order nodal divergence at a wall.  The control volume of a wall node is half a
// cell, 0 < n < h/2.  mlndlap_divu takes the tangential derivatives of the tangential
// velocity at the first cell centres, n = h/2, for the whole half volume; their mean over
// it is their value at n = h/4 to second order, which differs by (h/4) d/dn.  This adds,
// after the Neumann doubling, the linear extrapolation from the first two rows of cells:
// D(h/4) = 5/4 D(h/2) - 1/4 D(3h/2), i.e. the correction 1/4 [D(h/2) - D(3h/2)].  Nodes
// on two or more walls are left alone.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlndlap_divu_wall_corr (int i, int j, int k, Array4<Real> const& rhs,
                             Array4<Real const> const& vel, Array4<int const> const& msk,
                             GpuArray<Real,AMREX_SPACEDIM> const& dxinv, Box const& nddom,
                             GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                             GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc) noexcept
{
    if (msk(i,j,k)) { return; }
    IntVect const iv(AMREX_D_DECL(i,j,k));
    int side = 0;
    int const nd = nodelap_detail::mlndlap_single_wall(iv, nddom, lobc, hibc, side);
    if (nd < 0) { return; }
    Real corr = Real(0.0);
    for (int td = 0; td < AMREX_SPACEDIM; ++td) {
        if (td == nd) { continue; }
        Real D[2];
        for (int m = 0; m < 2; ++m) {
            int const cn = (side == 0) ? iv[nd] + m : iv[nd] - 1 - m;
            Real sum = Real(0.0);
            int cnt = 0;
#if (AMREX_SPACEDIM == 3)
            int const od = 3 - nd - td;
            for (int co = iv[od]-1; co <= iv[od]; ++co) {
#else
            {
#endif
                IntVect chi(iv), clo(iv);
                chi[nd] = cn; clo[nd] = cn;
                chi[td] = iv[td]; clo[td] = iv[td] - 1;
#if (AMREX_SPACEDIM == 3)
                chi[od] = co; clo[od] = co;
#endif
                sum += (vel(chi,td) - vel(clo,td)) * dxinv[td];
                ++cnt;
            }
            D[m] = sum / Real(cnt);
        }
        corr += Real(0.25) * (D[0] - D[1]);
    }
    rhs(i,j,k) += corr;
}

// The same for a cell-centred source: its half-cell mean is its value at n = h/4, while
// mlndlap_rhcc (after the Neumann doubling) uses the first-row mean, n = h/2.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlndlap_rhcc_wall_corr (int i, int j, int k, Array4<Real> const& rhs,
                             Array4<Real const> const& rhcc, Array4<int const> const& msk,
                             Box const& nddom,
                             GpuArray<LinOpBCType,AMREX_SPACEDIM> const& lobc,
                             GpuArray<LinOpBCType,AMREX_SPACEDIM> const& hibc) noexcept
{
    if (msk(i,j,k)) { return; }
    IntVect const iv(AMREX_D_DECL(i,j,k));
    int side = 0;
    int const nd = nodelap_detail::mlndlap_single_wall(iv, nddom, lobc, hibc, side);
    if (nd < 0) { return; }
    Real S[2];
    for (int m = 0; m < 2; ++m) {
        int const cn = (side == 0) ? iv[nd] + m : iv[nd] - 1 - m;
        Real sum = Real(0.0);
        int cnt = 0;
        // All cells of row m touching the node: 2 in 2D, 4 in 3D.
        IntVect c(iv);
        c[nd] = cn;
#if (AMREX_SPACEDIM == 3)
        int const t1 = (nd == 0) ? 1 : 0;
        int const t2 = (nd == 2) ? 1 : 2;
        for (int a = iv[t1]-1; a <= iv[t1]; ++a) {
            for (int b = iv[t2]-1; b <= iv[t2]; ++b) {
                c[t1] = a; c[t2] = b;
                sum += rhcc(c); ++cnt;
            }
        }
#else
        int const t1 = 1 - nd;
        for (int a = iv[t1]-1; a <= iv[t1]; ++a) {
            c[t1] = a;
            sum += rhcc(c); ++cnt;
        }
#endif
        S[m] = sum / Real(cnt);
    }
    rhs(i,j,k) += Real(0.25) * (S[0] - S[1]);
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
