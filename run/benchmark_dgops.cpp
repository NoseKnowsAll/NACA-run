#include "dg.h"
#include <ios>
#include <iostream>
#include <sstream>

const std::string suffix   = "_v2";
const int         order    = 3;
const int         refine   = 12;
const std::string meshdir  = "/scratch/mfranco/2021/naca/meshes/";
const std::string meshname = "naca" + suffix + "_p" + to_string(order) + "_r" + to_string(refine);
const std::string pre      = "/scratch/mfranco/2021/naca/run/results/" + meshname + "/";
// TODO: Technically hLE is a bit smaller than hwing, but vast majority of BL elements will be based on hwing size. Should we use hwing?
const double      hwing    = 0.049; // Make sure this is updated with correct value from naca_vX.geo
const double      Re       = 9.0*order/(hwing/(1<<refine)); // Because h/p = 10/Re sets safe h for boundary layer, this should be safe Re
const double      M0       = 0.25;
const double      AoAdeg   = 0.0;
const double      dt       = 1e-4;
const int         nsteps   = 5;   // Number of iterations to compute in order to avoid timing discrepancies
const int         presteps = 10;
const int         step0    = 1;   // set to 1 if you have precomputed solution
const int         writeint = 10;
const int         nstages  = 3;

const double      linerror = 1e-4;
const double      nlerror  = 1e-6;

enum DGOperations {
		   jac_assembly = 1,
		   res_eval     = 2,
		   mat_vec      = 3,
		   mass_inv     = 4
};

int main(int argc, char **argv) {
  
  MPI::Init(argc, argv);

  // Initialize data and physics of N-S simulation according to our parameters
  mesh msh;
  data d;
  phys p;
  
  int np = MPI::COMM_WORLD.Get_size();
  msh.readfile_mpi(meshdir+meshname, np);
  
  dginit(msh, d);
  
  double Reyn = Re;
  int wall = std::isfinite(Reyn) ? 2 : 3;
  dgprintf("Re = %f\n", Re);
  
  double AoA = AoAdeg*M_PI/180.0;
  using dg::physinit::FarFieldQty;
  dg::physinit::navierstokes(
      msh,
      {wall, 1},                // boundary conditions
      {Re, 0.72, 0.0, 1.0, 0.0},// pars: Re, Pr, C11 interior, C11 boundary, hscale with C11
      1.0,                      // far field density
      {cos(AoA), sin(AoA)},     // far field velocity
      FarFieldQty::Mach(M0),    // far field mach
      &p);
  p.viscous = true;

  // Initialize solver parameters
  int maxiter = 120;
  int restart = 30;
  auto linsolver = LinearSolverOptions::gmres("j", linerror, maxiter, restart); // j => Jacobi, i => ILU
  auto newton = NewtonOptions(linsolver, nlerror);
  
  // Set up the solution variable
  darray u;
  dgfreestream(msh, p, u);
  // Allocate space for internal arrays
  int N = u.size(1);
  darray r(d.ns, N, msh.nt);
  darray du(d.ns, N, msh.nt);
  jacarray Ddrdu(N*d.ns, N*d.ns, msh.nt);
  jacarray Odrdu(N*d.nes, N*d.ns, msh.nf, msh.nt);
  jacarray NULLARR;
						    
  // Initial steps to reduce transients
  const irk_scheme *rk11 = dirk_coeffs(1, 1); // Backward Euler
  if (step0 == 0) {
    dgprintf(" >>> Initial Steps <<<\n");
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 1.%d <<<\n", i);
      dgirktime(dgnavierstokes, u, msh, d, p, dt/100, 1, rk11, newton, "");
    }
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 2.%d <<<\n", i);
      dgirktime(dgnavierstokes, u, msh, d, p, dt/50, 1, rk11, newton, "");
    }
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 3.%d <<<\n", i);
      dgirktime(dgnavierstokes, u, msh, d, p, dt/20, 1, rk11, newton, "");
    }
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 4.%d <<<\n", i);
      dgirktime(dgnavierstokes, u, msh, d, p, dt/10, 1, rk11, newton, "");
    }
    fwritesolution(pre + "sol" + fill_int_to_string(1, 5, '0') + ".dat", u, msh);
  } else {
    // Don't compute initial steps. Load soln from file instead
    dgprintf(" >>> Checkpointing from %d <<<\n", step0);
    freadsolution(pre + "sol" + fill_int_to_string(step0, 5, '0') + ".dat", u, msh);
    
  }
  
  // Time DG operations
  for (int test = 1; test <= 4; test++) {
    DGOperations to_time = static_cast<DGOperations>(test);
    double min_time = 1.0/0.0;
    for (int i = 0; i < nsteps; i++) {
      set_timer();
      switch (to_time) {
      case jac_assembly:
	dgassemble(dgnavierstokes, u, r, Ddrdu, Odrdu, msh, d, p);
	break;
      case res_eval:
	dgassemble(dgnavierstokes, u, r, NULLARR, NULLARR, msh, d, p);
	break;
      case mat_vec:
	matvec(msh, d, msh.porder, Ddrdu, Odrdu, r, du);
	break;
      case mass_inv:
	dgmassinv(r, msh, d);
	break;
      }
      double time = get_timer();
      if (time < min_time)
	min_time = time;
    }
    printf("%d: %10.5f\n", test, min_time);
  }
  
  MPI::Finalize();
}
