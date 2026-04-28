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

#include "pgen/pgen.hpp"

// Non-relativistic Sedov blast wave.
// As descriged in in the Castro test suite
// https://amrex-astro.github.io/Castro/docs/Verification.html

namespace star_shredder {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(
      typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::FMKS),
      "Problem \"star_shredder\" requires \"FMKS\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v =
      rc->PackVariables({fluid_prim::density::name(), fluid_prim::velocity::name(),
                         fluid_prim::energy::name(), fluid_prim::bfield::name(),
                         fluid_prim::ye::name(), fluid_prim::pressure::name(),
                         fluid_prim::temperature::name(), fluid_prim::gamma1::name()},
                        imap);

  const int irho = imap[fluid_prim::density::name()].first;
  const int ivlo = imap[fluid_prim::velocity::name()].first;
  const int ivhi = imap[fluid_prim::velocity::name()].second;
  const int ieng = imap[fluid_prim::energy::name()].first;
  const int ib_lo = imap[fluid_prim::bfield::name()].first;
  const int ib_hi = imap[fluid_prim::bfield::name()].second;
  const int iye = imap[fluid_prim::ye::name()].second;
  const int iprs = imap[fluid_prim::pressure::name()].first;
  const int itmp = imap[fluid_prim::temperature::name()].first;
  const int igm1 = imap[fluid_prim::gamma1::name()].first;

  const Real rho_star = pin->GetOrAddReal("star_shredder", "rho_star", 1.0);
  const Real rstar = pin->GetOrAddReal("star_shredder", "rstar", 0.01);
  const Real separation = pin->GetOrAddReal("star_shredder", "sep", 12);
  const Real ringwidth = pin->GetOrAddReal("star_shredder", "ringwidth", 1.0);
  const Real ringtheta = pin->GetOrAddReal("star_shredder", "ringtheta", 1.57);
  const Real ringphi = pin->GetOrAddReal("star_shredder", "ringphi", 3.14);
  //const bool spherical = pin->GetOrAddBoolean("star_shredder", "spherical_coords", true);

  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  //Real Pa = pin->GetOrAddReal("star_shredder", "P_ambient", 1e-5);
  //Real Eexp = pin->GetOrAddReal("star_shredder", "explosion_energy", 1);

  //const Real v_inner = (4. / 3.) * M_PI * std::pow(rinner, 3.);
  //const Real uinner = Eexp / v_inner;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  // set up transformation stuff
  auto gpkg = pmb->packages.Get("geometry");
  bool derefine_poles = gpkg->Param<bool>("derefine_poles");
  Real h = gpkg->Param<Real>("h");
  Real xt = gpkg->Param<Real>("xt");
  Real alpha = gpkg->Param<Real>("alpha");
  Real x0 = gpkg->Param<Real>("x0");
  Real smooth = gpkg->Param<Real>("smooth");
  auto tr = Geometry::McKinneyGammieRyan(derefine_poles, h, xt, alpha, x0, smooth);

  pmb->par_for(
      "Phoebus::ProblemGenerator::star_shredder", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Coordinate transformations
        Real x1 = coords.Xc<1>(i);
        Real x2 = coords.Xc<2>(j);
        Real phi = coords.Xc<3>(k);
        Real r = tr.bl_radius(x1);
        Real th = tr.bl_theta(x1, x2);
        Real x = r * std::sin(th) * std::cos(phi);
        Real y = r * std::sin(th) * std::sin(phi);
        Real z = r * std::cos(th);
        // Get the x,y,z coordinates of the blob
        Real xblob = separation * std::sin(ringtheta) * std::cos(ringphi);
        Real yblob = separation * std::sin(ringtheta) * std::sin(ringphi);
        Real zblob = separation * std::cos(ringtheta);

        // Density
        Real rho = 1e-8;
        Real P = 1e-11;
        Real lambda[2];
        if ((ndim == 3) && (std::sqrt(
            (x-xblob) * (x-xblob) + 
            (y-yblob) * (y-yblob) + 
            (z-zblob) * (z-zblob)
        ) < ringwidth*ringwidth)) {
            rho = rho_star;
            P = rho_star;
        } else if (ndim == 2)
        {
            Real rcyl = std::sqrt(x*x + y*y);
            Real rcyl_blob = std::sqrt(xblob*xblob + yblob*yblob);
            if (((rcyl-rcyl_blob)*(rcyl-rcyl_blob) + (z-zblob)*(z-zblob))< ringwidth*ringwidth) {
                rho = rho_star;
                P = rho_star;
            }
        }
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          lambda[0] = v(iye, k, j, i);
        }

        const Real u = phoebus::energy_from_rho_P(eos, rho, P, emin, emax, lambda[0]);

        const Real eps = u / (rho + 1e-20);
        const Real T = eos.TemperatureFromDensityInternalEnergy(rho, eps, lambda);
        //const Real P = eos.PressureFromDensityInternalEnergy(rho, eps, lambda);

        v(irho, k, j, i) = rho;
        v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = u;
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);
        for (int d = ivlo; d <= ivhi; d++)
          v(d, k, j, i) = 0.0;
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace star_shredder
