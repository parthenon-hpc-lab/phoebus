#include <memory>
#include <vector>
#include <cstdio>

#include "analysis/analysis.hpp"
#include "analysis/history.hpp"
#include "ascii_reader.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "progenitordata.hpp"
namespace Progenitor {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto progenitor_pkg = std::make_shared<StateDescriptor>("progenitor");
  Params &params = progenitor_pkg->AllParams();

  bool enabled = pin->GetOrAddBoolean("progenitor", "enabled", false);
  params.Add("enabled", enabled);
  if (!enabled) return progenitor_pkg;

  // Read the table
  std::vector<double> mydata[11];
  const std::string table = pin->GetString("progenitor", "tablepath");
  AsciiReader::readtable(table, mydata);
  const int npoints = mydata[0].size();

  // DataBoxes for interpolation functions
  Spiner::DataBox<Real> r(npoints);
  Spiner::DataBox<Real> mass_density(npoints);
  Spiner::DataBox<Real> temp(npoints);
  Spiner::DataBox<Real> Ye(npoints);
  Spiner::DataBox<Real> specific_internal_energy(npoints);
  Spiner::DataBox<Real> velocity(npoints);
  Spiner::DataBox<Real> pressure(npoints);

  Spiner::DataBox<Real> adm_density(npoints);
  Spiner::DataBox<Real> adm_momentum(npoints);
  Spiner::DataBox<Real> S_adm(npoints);
  Spiner::DataBox<Real> Srr_adm(npoints);

  // Fill in the variables
  // Requires same grid for primitive and adm quantities in the input table
  for (int i = 0; i < npoints; ++i) {
    r(i) = mydata[AsciiReader::COLUMNS::rad][i];

    mass_density(i) = mydata[AsciiReader::COLUMNS::rho][i];
    temp(i) = mydata[AsciiReader::COLUMNS::temp][i];
    Ye(i) = mydata[AsciiReader::COLUMNS::ye][i];
    specific_internal_energy(i) = mydata[AsciiReader::COLUMNS::eps][i];
    velocity(i) = mydata[AsciiReader::COLUMNS::vel][i];
    pressure(i) = mydata[AsciiReader::COLUMNS::press][i];

    adm_density(i) = mydata[AsciiReader::COLUMNS::rho_adm][i];
    adm_momentum(i) = mydata[AsciiReader::COLUMNS::P_adm][i];
    S_adm(i) = mydata[AsciiReader::COLUMNS::S_adm][i];
    Srr_adm(i) = mydata[AsciiReader::COLUMNS::Srr_adm][i];
  } // for loop (i)

  // Interpolation functions
  mass_density.setRange(0, r(0), r(npoints - 1), npoints);
  temp.setRange(0, r(0), r(npoints - 1), npoints);
  Ye.setRange(0, r(0), r(npoints - 1), npoints);
  specific_internal_energy.setRange(0, r(0), r(npoints - 1), npoints);
  velocity.setRange(0, r(0), r(npoints - 1), npoints);
  pressure.setRange(0, r(0), r(npoints - 1), npoints);

  adm_density.setRange(0, r(0), r(npoints - 1), npoints);
  adm_momentum.setRange(0, r(0), r(npoints - 1), npoints);
  S_adm.setRange(0, r(0), r(npoints - 1), npoints);
  Srr_adm.setRange(0, r(0), r(npoints - 1), npoints);

  // Get on device
  auto mass_density_dev = mass_density.getOnDevice();
  auto temp_dev = temp.getOnDevice();
  auto Ye_dev = Ye.getOnDevice();
  auto specific_internal_energy_dev = specific_internal_energy.getOnDevice();
  auto velocity_dev = velocity.getOnDevice();
  auto pressure_dev = pressure.getOnDevice();

  auto adm_density_dev = adm_density.getOnDevice();
  auto adm_momentum_dev = adm_momentum.getOnDevice();
  auto S_adm_dev = S_adm.getOnDevice();
  auto Srr_adm_dev = Srr_adm.getOnDevice();

  // Post-processing params
  Real outside_pns_threshold =
      pin->GetOrAddReal("progenitor", "outside_pns_threshold",
                        2.42e-5); // corresponds to entropy > 3 kb/baryon
  Real inside_pns_threshold = pin->GetOrAddReal("progenitor", "inside_pns_threshold",
                                                0.008); // corresponds to r < 80 km
  Real net_heat_threshold = pin->GetOrAddReal("progenitor", "net_heat_threshold",
                                              1e-8); // corresponds to r < 80 km
  auto mdot_radii = pin->GetOrAddVector("progenitor", "mdot_radii",
                                        std::vector<Real>{400}); // default 400km

  // unit conversions
  UnitConversions units(pin);
  CodeConstants consts(units);
  Real LengthCGSToCode = units.GetLengthCGSToCode();
  Real DensityCGSToCode = units.GetMassDensityCGSToCode();
  Real EntropyCGSToCode = units.GetEntropyCGSToCode();
  // assume kb/baryon ~ kb/proton mass --> erg/g/K (specific entropy)
  Real EntropykBToCGS = consts.kb / consts.mp; 

  // Add Params
  params.Add("mass_density", mass_density);
  params.Add("temp", temp);
  params.Add("Ye", Ye);
  params.Add("specific_internal_energy", specific_internal_energy);
  params.Add("velocity", velocity);
  params.Add("pressure", pressure);

  params.Add("adm_density", adm_density);
  params.Add("adm_momentum", adm_momentum);
  params.Add("S_adm", S_adm);
  params.Add("Srr_adm", Srr_adm);

  params.Add("mass_density_dev", mass_density_dev);
  params.Add("temp_dev", temp_dev);
  params.Add("Ye_dev", Ye_dev);
  params.Add("specific_internal_energy_dev", specific_internal_energy_dev);
  params.Add("velocity_dev", velocity_dev);
  params.Add("pressure_dev", pressure_dev);

  params.Add("adm_density_dev", adm_density_dev);
  params.Add("adm_momentum_dev", adm_momentum_dev);
  params.Add("S_adm_dev", S_adm_dev);
  params.Add("Srr_adm_dev", Srr_adm_dev);

  // criterion for bounce in ccsne (O'Connor & Ott 2010)
  params.Add("bounce_density", Constants::BOUNCE_DENS * DensityCGSToCode);
  params.Add("bounce_entropy", Constants::BOUNCE_ENTR * EntropykBToCGS * EntropyCGSToCode);
  // set these if bounce occurs
  params.Add("bounce_density_actual"; -1.0);
  params.Add("bounce_entropy_actual"; -1.0);
  params.Add("bounce_time"; -1.0);
  params.Add("post_bounce", False);

  params.Add("outside_pns_threshold", outside_pns_threshold);
  params.Add("inside_pns_threshold", inside_pns_threshold);
  params.Add("net_heat_threshold", net_heat_threshold);
  params.Add("mdot_radii", mdot_radii);

  // Reductions
  auto HstSum = parthenon::UserHistoryOperation::sum;
  auto HstMax = parthenon::UserHistoryOperation::max;
  using History::ReduceInGain;
  using History::ReduceOneVar;
  using parthenon::HistoryOutputVar;
  parthenon::HstVar_list hst_vars = {};
  auto Mgain = [](MeshData<Real> *md) {
    return ReduceInGain<fluid_cons::density>(md, 1, 0);
  };
  auto Qgain = [](MeshData<Real> *md) {
    return ReduceInGain<internal_variables::GcovHeat>(md, 0, 0) -
           ReduceInGain<internal_variables::GcovCool>(md, 0, 0);
  };
  for (auto rc : mdot_radii) {
    auto rc_code = rc * 1e5 * LengthCGSToCode;
    auto Mdot = [rc_code](MeshData<Real> *md) {
      return History::CalculateMdot(md, rc_code, false);
    };
    hst_vars.emplace_back(
        HistoryOutputVar(HstSum, Mdot, "Mdot at r = " + std::to_string(int(rc)) + "km"));
  }

  Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  auto Mdot_gain = [x1max](MeshData<Real> *md) {
    return History::CalculateMdot(md, x1max, true);
  };

  const bool rad_active = pin->GetBoolean("physics", "rad");
  if (rad_active) {
    const bool do_gain_calc = pin->GetOrAddBoolean("radiation", "do_gain_calc", false);
    if (do_gain_calc) {
      hst_vars.emplace_back(HistoryOutputVar(HstSum, Mgain, "Mgain"));
      hst_vars.emplace_back(HistoryOutputVar(HstSum, Qgain, "total net heat"));
      hst_vars.emplace_back(HistoryOutputVar(HstSum, Mdot_gain, "Mdot gain"));
    }
  }
  params.Add(parthenon::hist_param_key, hst_vars);

  return progenitor_pkg;
} // Initialize

TaskStatus GetProgenitorState(MeshData<Real> *md, Real simtime) {

	// bounds
	const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

	// namespaces, etc.
	using parthenon::MakePackDescriptor;
	namespace p = fluid_prim;
	auto *pmb = md->GetParentPointer();
  Mesh *pmesh = md->GetMeshPointer();
	
	// could re-calc entropy if needed, but requires an extra eos retrieval + call
	auto &resolved_pkgs = pmesh->resolved_packages;
	static auto desc = MakePackDescriptor<p::density, p::entropy>(resolved_pkgs.get());
  
  auto v = desc.GetPack(md);
  const int nblocks = v.GetNBlocks();

	// reading in parameters/conditions from package
	auto progen = pmb->packages.Get("progenitor").get(); // actual progenitor package
  // todo: add conversions to code units here or above at param init.
	const Real bounce_density_crit = progen->Param<Real>("bounce_density");
	const Real bounce_entropy_crit = progen->Param<Real>("bounce_entropy");

	Real max_density, min_entropy;
	bool post_bounce = progen->Param<bool>("post_bounce");
  typename Kokkos::MinMax<Real>::value_type minmax; // is this right?

  parthenon::par_reduce( 
    parthenon::LoopPatternMDRange(), "Calculates max density and min entropy (pre-bounce) in SNe.",
    DevExecSpace(), 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, 
		KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, typename Kokkos::MinMAx<Real>::value_type &lminmax) {
			// checking for maximum density and minimum entropy
      lminmax.min_val = ( v(b, p::entropy(), k, j, i) < lminmax.min_val ? v(b, p::entropy(), k, j, i) : lminmax.min_val );
      lminmax.max_val = ( v(b, p::density(), k, j, i) > lminmax.max_val ? v(b, p::density(), k, j, i) : lminmax.max_val );
		},
		Kokkos::MinMax<Real>(minmax)
  );

  max_density = minmax.max_val;
  min_entropy = minmax.min_val;

	if ( max_density >= bounce_density_crit || min_entropy <= bounce_entropy_crit) {
		
    post_bounce = true; // bounce reached!
		bounce_time = simtime; // capture bounce time

    // update in params for continuity (also in case of restart?)
    progen->Param<bool>("post_bounce") = post_bounce;
    progen->Param<Real>("bounce_time") = bounce_time;
		// set bounce params for density, entropy
    progen->Param<Real>("bounce_density_actual") = max_density;
    progen->Param<Real>("bounce_entropy_actual") = min_entropy;

 		// output data file, in code units
    FILE *fout;
    fout = fopen("bounce.dat", "w"); // questionable naming...
    fprintf("%30s\n", ">> bounce reached!");
    fprintf("%30s  %.14e\n", "bounce time", bounce_time);
    fprintf("%30s  %.14e\n", "bounce density", max_density);
    fprintf("%30s  %.14e\n", "bounce entropy", min_entropy);
    fclose(fout);

  }

  // todo: do we want a bounce.dat file even if bounce isn't reached? just for progen stats?
  
  return TaskStatus::complete; // can this return void?? does status matter in this case

} // GetProgenitorState

} // namespace Progenitor
