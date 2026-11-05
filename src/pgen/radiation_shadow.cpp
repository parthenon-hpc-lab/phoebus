// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
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

#include <cmath>
#include <string>

#include "pgen/pgen.hpp"
#include "phoebus_utils/programming_utils.hpp"

namespace radiation_shadow {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::Minkowski),
                    "Problem \"advection\" requires \"Minkowski\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {radmoment_prim::J::name(), radmoment_prim::H::name(),
       radmoment_cons::E::name(), radmoment_cons::F::name(),
      radmoment_internal::kappaH::name(), radmoment_internal::kappaJ::name(),
       radmoment_internal::xi::name(), radmoment_internal::phi::name(),
       fluid_prim::density::name(), fluid_prim::temperature::name(),
       fluid_prim::pressure::name(), fluid_prim::gamma1::name(),
       fluid_prim::energy::name(), fluid_prim::ye::name(), fluid_prim::velocity::name()},
      imap);

  auto idJ = imap.GetFlatIdx(radmoment_prim::J::name());
  auto idH = imap.GetFlatIdx(radmoment_prim::H::name());
  auto idE = imap.GetFlatIdx(radmoment_cons::E::name());
  auto idF = imap.GetFlatIdx(radmoment_cons::F::name());
  auto idkappaJ = imap.GetFlatIdx(radmoment_internal::kappaJ::name());
  auto idkappaH = imap.GetFlatIdx(radmoment_internal::kappaH::name());
  auto idv = imap.GetFlatIdx(fluid_prim::velocity::name());
  auto ixi = imap.GetFlatIdx(radmoment_internal::xi::name());
  auto iphi = imap.GetFlatIdx(radmoment_internal::phi::name());
  const int iRho = imap[fluid_prim::density::name()].first;
  const int iT = imap[fluid_prim::temperature::name()].first;
  const int iP = imap[fluid_prim::pressure::name()].first;
  const int igm1 = imap[fluid_prim::gamma1::name()].first;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int pye = imap[fluid_prim::ye::name()].first;
  Params &phoebus_params = pmb->packages.Get("phoebus")->AllParams();

  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");

  const Real rho    = pin->GetOrAddReal("radiation_shadow", "rho", 1.0);
  const Real kap_a  = pin->GetOrAddReal("radiation_shadow", "kap_a", 0.);
  const Real kap_s  = pin->GetOrAddReal("radiation_shadow", "kap_s", 0.);

  const Real R_star = pin->GetOrAddReal("radiation_shadow", "R_star", 1.0);

  const auto specB = idJ.GetBounds(1);
  const Real vx = pin->GetOrAddReal("radiation_shadow", "vx", 1e-5);
  PARTHENON_REQUIRE(std::fabs(vx) < 1., "Subluminal velocity required!");

  auto unit_conv = phoebus::UnitConversions(pin);
  const Real rho0 = 1. * unit_conv.GetMassDensityCodeToCGS();

  const Real dx1 = (pin->GetReal("parthenon/mesh", "x1max") -
                    pin->GetReal("parthenon/mesh", "x1min")) *
                   unit_conv.GetLengthCodeToCGS();
  const Real xmin = pin->GetReal("parthenon/mesh", "x1min");

  // Optical depth over the entire X1 simulation region
  const Real tau = pin->GetOrAddReal("radiation_shadow", "tau", 1.e3);

  const Real kappa = tau / (rho0 * dx1);

  pin->SetReal("s_opacity", "gray_kappa", kappa);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const int nghost = pin->GetInteger("parthenon/mesh", "nghost");

  Real T = 1.;

  auto rad = pmb->packages.Get("radiation").get();
  auto species = rad->Param<std::vector<Microphysics::RadiationType>>("species");
  auto num_species = rad->Param<int>("num_species");

  const Real W = 1 / sqrt(1 - vx * vx);
  pmb->par_for(
      "Phoebus::ProblemGenerator::radiation_shadow", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real r = std::sqrt(x * x + y * y);
        const Real dx = coords.Dxc<1>(k, j, i);

        const Real P = eos.PressureFromDensityTemperature(rho, T);
        const Real eps = eos.InternalEnergyFromDensityTemperature(rho, T);

        v(iRho, k, j, i) = rho;
        v(iT, k, j, i) = T;
        v(iP, k, j, i) = P;
        v(ieng, k, j, i) = v(iRho, k, j, i) * eps;
        v(igm1, k, j, i) =
            eos.BulkModulusFromDensityTemperature(v(iRho, k, j, i), v(iT, k, j, i)) /
            v(iP, k, j, i);
        v(pye, k, j, i) = 0.5;

        v(idv(0), k, j, i) = W * vx;
        v(idv(1), k, j, i) = 1e-5;
        v(idv(2), k, j, i) = 1e-5;

        for (int ispec = specB.s; ispec <= specB.e; ++ispec) {
          v(ixi(ispec), k, j, i) = 1.0;
          v(iphi(ispec), k, j, i) = acos(-1.0) * 1.000001;

          if (r > R_star) {
          v(idkappaJ(ispec), k, j, i) = 1e-5;
          v(idkappaH(ispec), k, j, i) = 1e-5;
          } else {
            v(idkappaJ(ispec), k, j, i) = kap_a;
            v(idkappaH(ispec), k, j, i) = kap_s + kap_a;
          }

          // const bool inject_beam = (i < nghost) && (x < xmin) && (y < 1) && (y > -1);
          const bool inject_beam = (x < -1.95 && x > -2.0) && (y < 1) && (y > -1);
          if (inject_beam) {
              v(idE(ispec), k, j, i)    = 1.0;
              v(idF(ispec, 0), k, j, i) = 1.0;
              v(idF(ispec, 1), k, j, i) = 1e-5;
              v(idF(ispec, 2), k, j, i) = 1e-5;

              v(idJ(ispec), k, j, i)    = 1.0;
              v(idH(ispec, 0), k, j, i) = 9.990000e-01;
              v(idH(ispec, 1), k, j, i) = 1e-5;
              v(idH(ispec, 2), k, j, i) = 1e-5;
          } else {
              v(idE(ispec), k, j, i)    = 1e-5;
              v(idF(ispec, 0), k, j, i) = 1e-5;
              v(idF(ispec, 1), k, j, i) = 1e-5;
              v(idF(ispec, 2), k, j, i) = 1e-5;

              v(idJ(ispec), k, j, i) = 1e-5;
              v(idH(ispec, 0), k, j, i) = 1e-5;
              v(idH(ispec, 1), k, j, i) = 1e-5;
              v(idH(ispec, 2), k, j, i) = 1e-5;
          }
        }
      });
  radiation::MomentCon2Prim(rc.get());
  radiation::MomentPrim2Con(rc.get(), IndexDomain::interior);
}

void PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, SimTime &sim_time) {
  
  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    PackIndexMap imap;
    auto v = rc->PackVariables(
        std::vector<std::string>(
            {radmoment_prim::J::name(), radmoment_prim::H::name(),
            radmoment_cons::E::name(), radmoment_cons::F::name(),
            radmoment_internal::kappaH::name(), radmoment_internal::kappaJ::name(),
            fluid_prim::density::name(), fluid_prim::temperature::name(),
            fluid_prim::energy::name(), fluid_prim::velocity::name(),
            radmoment_internal::xi::name(), radmoment_internal::phi::name()}),
        imap);

    auto idJ = imap.GetFlatIdx(radmoment_prim::J::name());
    auto idH = imap.GetFlatIdx(radmoment_prim::H::name());
    auto idE = imap.GetFlatIdx(radmoment_cons::E::name());
    auto idF = imap.GetFlatIdx(radmoment_cons::F::name());
    auto idkappaJ = imap.GetFlatIdx(radmoment_internal::kappaJ::name());
    auto idkappaH = imap.GetFlatIdx(radmoment_internal::kappaH::name());
    auto idv = imap.GetFlatIdx(fluid_prim::velocity::name());
    auto ixi = imap.GetFlatIdx(radmoment_internal::xi::name());
    auto iphi = imap.GetFlatIdx(radmoment_internal::phi::name());
    const int prho = imap[fluid_prim::density::name()].first;
    const int pT = imap[fluid_prim::temperature::name()].first;
    const int peng = imap[fluid_prim::energy::name()].first;

    Params &phoebus_params = pmb->packages.Get("phoebus")->AllParams();

    const Real rho    = pin->GetReal("radiation_shadow", "rho");
    const Real kap_a  = pin->GetReal("radiation_shadow", "kap_a");
    const Real kap_s  = pin->GetReal("radiation_shadow", "kap_s");

    const Real R_star = pin->GetReal("radiation_shadow", "R_star");

    const auto specB = idJ.GetBounds(1);

    auto &coords = pmb->coords;
    auto pmesh = pmb->pmy_mesh;
    int ndim = pmesh->ndim;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

    const int nghost = pin->GetInteger("parthenon/mesh", "nghost");
    const Real xmin = pin->GetReal("parthenon/mesh", "x1min");

    auto rad = pmb->packages.Get("radiation").get();
    auto species = rad->Param<std::vector<Microphysics::RadiationType>>("species");
    auto num_species = rad->Param<int>("num_species");

    pmb->par_for(
        "Phoebus::ProblemGenerator::radiation_shadow", kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real r = std::sqrt(x * x + y * y);
          const Real dx = coords.Dxc<1>(k, j, i);

          for (int ispec = specB.s; ispec <= specB.e; ++ispec) {
            if (r > R_star) {
            v(idkappaJ(ispec), k, j, i) = 1e-5;
            v(idkappaH(ispec), k, j, i) = 1e-5;
            } else {
              v(idkappaJ(ispec), k, j, i) = 1.0;
              v(idkappaH(ispec), k, j, i) = 1.0;
            }

            // const bool inject_beam = (i < nghost) && (x < xmin) && (y < 1) && (y > -1);
            const bool inject_beam = (x < -1.9 && x > -2.0) && (y < 1) && (y > -1);
            if (inject_beam) {
              const Real fct = 1e0;
              v(idE(ispec), k, j, i)    = 1.0;
              v(idF(ispec, 0), k, j, i) = 1.0;
              v(idF(ispec, 1), k, j, i) = 1e-5;
              v(idF(ispec, 2), k, j, i) = 1e-5;

              v(idJ(ispec), k, j, i)    = 1.0;
              v(idH(ispec, 0), k, j, i) = 9.990000e-01;
              v(idH(ispec, 1), k, j, i) = 1e-5;
              v(idH(ispec, 2), k, j, i) = 1e-5;
            }
          }
        });

    radiation::MomentCon2Prim(rc.get());
    radiation::MomentPrim2Con(rc.get(), IndexDomain::interior);
  }
}
} // namespace radiation_shadow
