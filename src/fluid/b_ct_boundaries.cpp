// © 2026. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

// Physical boundary treatments for face-centered B and EMFs.

#include "b_ct.hpp"

#include <array>
#include <vector>

#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/robust.hpp"
#include "phoebus_utils/variables.hpp"

#include <kokkos_abstraction.hpp>

namespace b_ct {

namespace {

using TE = parthenon::TopologicalElement;

// Direction (1, 2, or 3) normal to a domain boundary face.
int BoundaryDirection(IndexDomain domain) {
  switch (domain) {
  case IndexDomain::inner_x1:
  case IndexDomain::outer_x1:
    return 1;
  case IndexDomain::inner_x2:
  case IndexDomain::outer_x2:
    return 2;
  default:
    return 3;
  }
}

bool BoundaryIsInner(IndexDomain domain) {
  return domain == IndexDomain::inner_x1 || domain == IndexDomain::inner_x2 ||
         domain == IndexDomain::inner_x3;
}

// Edges tangent to a boundary face.
std::array<TE, 2> TangentialEdges(int bdir) {
  if (bdir == 1) return {TE::E2, TE::E3};
  if (bdir == 2) return {TE::E1, TE::E3};
  return {TE::E1, TE::E2};
}

std::array<TE, 3> Faces() { return {TE::F1, TE::F2, TE::F3}; }

CellLocation FaceLocation(TE el) {
  if (el == TE::F1) return CellLocation::Face1;
  if (el == TE::F2) return CellLocation::Face2;
  return CellLocation::Face3;
}

TE FaceNormalTo(int bdir) {
  if (bdir == 1) return TE::F1;
  if (bdir == 2) return TE::F2;
  return TE::F3;
}

} // namespace

void OutflowFaceB(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse) {
  auto pmb = rc->GetBlockPointer();
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd")) return;

  auto Bf = rc->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  if (Bf.GetDim(4) == 0) return;
  auto geom = Geometry::GetCoordinateSystem(rc);
  const int bdir = BoundaryDirection(domain);
  const bool inner = BoundaryIsInner(domain);

  // Copy physical B into ghost faces.
  for (auto el : Faces()) {
    const auto ib = rc->GetBoundsI(IndexDomain::interior, el);
    const auto jb = rc->GetBoundsJ(IndexDomain::interior, el);
    const auto kb = rc->GetBoundsK(IndexDomain::interior, el);
    const int iref = (bdir == 1) ? (inner ? ib.s : ib.e) : 0;
    const int jref = (bdir == 2) ? (inner ? jb.s : jb.e) : 0;
    const int kref = (bdir == 3) ? (inner ? kb.s : kb.e) : 0;
    const auto loc = FaceLocation(el);

    pmb->par_for_bndry(
        "b_ct::OutflowFaceB", IndexRange{0, 0}, domain, el, coarse, /*fine=*/false,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const int kk = (bdir == 3) ? kref : k;
          const int jj = (bdir == 2) ? jref : j;
          const int ii = (bdir == 1) ? iref : i;
          const Real detg = geom.DetGamma(loc, k, j, i);
          const Real detg_ref = geom.DetGamma(loc, kk, jj, ii);
          Bf(el, 0, k, j, i) = robust::ratio(detg, detg_ref) * Bf(el, 0, kk, jj, ii);
        });
  }
}

void ReflectFaceB(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse) {
  auto pmb = rc->GetBlockPointer();
  if (!pmb->packages.Get("fluid")->Param<bool>("mhd")) return;

  auto Bf = rc->PackVariables(std::vector<std::string>{fluid_cons::fbfield::name()});
  if (Bf.GetDim(4) == 0) return;
  auto geom = Geometry::GetCoordinateSystem(rc);
  const int bdir = BoundaryDirection(domain);
  const bool inner = BoundaryIsInner(domain);
  const auto normal_face = FaceNormalTo(bdir);

  // Zero normal B; mirror normal/tangential fields with odd/even parity.
  for (auto el : Faces()) {
    const auto ib = rc->GetBoundsI(IndexDomain::interior, el);
    const auto jb = rc->GetBoundsJ(IndexDomain::interior, el);
    const auto kb = rc->GetBoundsK(IndexDomain::interior, el);
    const bool normal = el == normal_face;
    const int iref = (bdir == 1) ? (inner ? ib.s : ib.e) : 0;
    const int jref = (bdir == 2) ? (inner ? jb.s : jb.e) : 0;
    const int kref = (bdir == 3) ? (inner ? kb.s : kb.e) : 0;
    const int ioffset = 2 * iref + (normal ? 0 : (inner ? -1 : 1));
    const int joffset = 2 * jref + (normal ? 0 : (inner ? -1 : 1));
    const int koffset = 2 * kref + (normal ? 0 : (inner ? -1 : 1));
    const auto loc = FaceLocation(el);

    if (normal) {
      if (bdir == 1) {
        pmb->par_for(
            "b_ct::ReflectFaceBZeroNormal", kb.s, kb.e, jb.s, jb.e, iref, iref,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
              Bf(el, 0, k, j, i) = 0.0;
            });
      } else if (bdir == 2) {
        pmb->par_for(
            "b_ct::ReflectFaceBZeroNormal", kb.s, kb.e, jref, jref, ib.s, ib.e,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
              Bf(el, 0, k, j, i) = 0.0;
            });
      } else {
        pmb->par_for(
            "b_ct::ReflectFaceBZeroNormal", kref, kref, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
              Bf(el, 0, k, j, i) = 0.0;
            });
      }
    }
    pmb->par_for_bndry(
        "b_ct::ReflectFaceB", IndexRange{0, 0}, domain, el, coarse, /*fine=*/false,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const int kk = (bdir == 3) ? koffset - k : k;
          const int jj = (bdir == 2) ? joffset - j : j;
          const int ii = (bdir == 1) ? ioffset - i : i;
          const Real detg = geom.DetGamma(loc, k, j, i);
          const Real detg_ref = geom.DetGamma(loc, kk, jj, ii);
          const Real sign = normal ? -1.0 : 1.0;
          Bf(el, 0, k, j, i) =
              sign * robust::ratio(detg, detg_ref) * Bf(el, 0, kk, jj, ii);
        });
  }
}

void ZeroBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain) {
  auto pmb = rc->GetBlockPointer();
  auto emf =
      rc->PackVariables(std::vector<std::string>{internal_variables::eemf::name()});
  const int bdir = BoundaryDirection(domain);
  for (auto el : TangentialEdges(bdir)) {
    pmb->par_for_bndry(
        "b_ct::ZeroBoundaryEMF", IndexRange{0, 0}, domain, el, /*coarse=*/false,
        /*fine=*/false,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          emf(el, 0, k, j, i) = 0.0;
        });
  }
}

void AverageBoundaryEMF(MeshBlockData<Real> *rc, IndexDomain domain) {
  auto pmb = rc->GetBlockPointer();
  auto emf =
      rc->PackVariables(std::vector<std::string>{internal_variables::eemf::name()});
  const int ndim = pmb->pmy_mesh->ndim;
  const int bdir = BoundaryDirection(domain);
  const bool inner = BoundaryIsInner(domain);

  for (auto el : TangentialEdges(bdir)) {
    // In 2D there is no true E1 (X2 boundary) / E2 (X1 boundary) edge to average.
    if (ndim < 3 && ((bdir == 2 && el == TE::E1) || (bdir == 1 && el == TE::E2))) {
      continue;
    }

    IndexRange ib = rc->GetBoundsI(IndexDomain::interior, el);
    IndexRange jb = rc->GetBoundsJ(IndexDomain::interior, el);
    IndexRange kb = rc->GetBoundsK(IndexDomain::interior, el);
    const int iref = (bdir == 1) ? (inner ? ib.s : ib.e) : 0;
    const int jref = (bdir == 2) ? (inner ? jb.s : jb.e) : 0;
    const int kref = (bdir == 3) ? (inner ? kb.s : kb.e) : 0;

    pmb->par_for_bndry(
        "b_ct::AverageBoundaryEMF", IndexRange{0, 0}, domain, el, /*coarse=*/false,
        /*fine=*/false,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const int kk = (bdir == 3) ? kref : k;
          const int jj = (bdir == 2) ? jref : j;
          const int ii = (bdir == 1) ? iref : i;
          emf(el, 0, k, j, i) = emf(el, 0, kk, jj, ii);
        });
  }
}

} // namespace b_ct
