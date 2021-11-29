#include "dg.h"
#include "dgavtools.h"
#include <ios>
#include <iostream>
#include <sstream>

const std::string suffix   = "_v2";
const int         order    = 3;
const int         refine   = 12; // amount boundary layer has been refined
const int         R        = 30; // far field size
const std::string meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
//const std::string meshdir  = "/scratch/mfranco/2021/naca/meshes/";
const std::string meshname = "naca" + suffix + "_p" + to_string(order) + "_r" + to_string(refine) + "_R" + to_string(R);
const std::string pre      = "/scratch/mfranco/2021/naca/run/results/" + meshname + "/snaps/";
// TODO: Technically hLE is a bit smaller than hwing, but vast majority of BL elements will be based on hwing size. Should we use hwing?
const double      hwing    = 0.049; // Must be value from meshes/scripts/mk_naca_v*.m
const double      Re       = 9.0*order/(hwing/(1<<refine)); // Because h/p = 10/Re sets safe h for boundary layer, this should be safe Re
const double      M0       = 0.25;
const double      AoAdeg   = 0.0;
const double      dt       = 2e-4;
const double      Tfinal   = 0.0;  // define as 0 to use nsteps variable instead
const int         nsteps   = 5000; // only used if Tfinal == 0.0
const int         presteps = 10;
const int         step0    = 0;   // set to 1 if you have precomputed solution
const int         writeint = 50;
const int         nstages  = 3;

const double      linerror = 1e-4;
const double      nlerror  = 1e-6;

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
  int maxiter = 200;
  int restart = 30;
  auto linsolver = LinearSolverOptions::gmres("j", linerror, maxiter, restart); // j => Jacobi, i => ILU
  auto newton = NewtonOptions(linsolver, nlerror);
  
  // Set up the solution variable
  darray u;
  dgfreestream(msh, p, u);
  
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
    p.time = 0+step0*dt;
  }
    
  // Main loop
  const irk_scheme *dirk = dirk_coeffs(nstages,1); // L-stable, order == stages, DIRK
  int final_step = ( Tfinal == 0.0 ? step0+nsteps : int(round(Tfinal / dt)) );
  for (int i=step0+1; i<=final_step; i++) {
    dgprintf("\n >>> Step %5d <<< \n", i);
    dgirktime(dgnavierstokes, u, msh, d, p, dt, 1, dirk, newton, "");
    if (i % writeint == 0)
      fwritesolution(pre + "sol" + fill_int_to_string(i, 5, '0') + ".dat", u, msh);
  }
  fwritesolution(pre + "sol" + fill_int_to_string(final_step, 5, '0') + ".dat", u, msh);
  
  MPI::Finalize();
}
