#include "dg.h"
#include "dgavtools.h"
#include <ios>
#include <iostream>
#include <sstream>

const std::string suffix   = "_v1_p3";
const std::string meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
const std::string meshname = "naca" + suffix;
const std::string pre      = "/scratch/mfranco/2021/naca/run/results/" + meshname + "/";
const double      Re       = 3e6;
const double      M0       = 0.25;
const double      AoAdeg   = 0.0;
const double      dt       = 1e-4;
const double      Tfinal   = 0.0; // define as 0 to use nsteps variable instead
const int         nsteps   = 3;   // only used if Tfinal == 0.0
const int         presteps = 10;
const int         step0    = 0;
const int         writeint = 10;
const double      linerror = 1e-4;
const double      nlerror  = 1e-6;

int main(int argc, char **argv) {
  
  MPI::Init(argc, argv);
  
  mesh msh;
  data d;
  phys p;
  
  int np = MPI::COMM_WORLD.Get_size();
  int nstages = 2;
  int order = 2*nstages-1;
  msh.readfile_mpi(meshdir+meshname, np);
  
  dginit(msh, d);
  
  double Reyn = Re;
  int wall = std::isfinite(Reyn) ? 2 : 3;
  dgprintf("Wall boundary condition: %d\n", wall);
  
  double AoA = AoAdeg*M_PI/180.0;
  using dg::physinit::FarFieldQty;
  dg::physinit::navierstokes(
      msh,
      {wall, 1},                // bndcnds
      {Re, 0.72, 0.0, 1.0, 0.0},// pars
      1.0,                      // far field density
      {cos(AoA), sin(AoA)},     // far field velocity
      FarFieldQty::Mach(M0),    // far field mach
      &p);
  p.viscous = true;

  int maxiter = 120;
  int restart = 30;
  auto linsolver = LinearSolverOptions::gmres("i", linerror, maxiter, restart);
  auto newton = NewtonOptions(linsolver, nlerror);
  
  // Set up the solution variable (add extra component into u for viscosity)
  darray u, u0, u1(msh.ns, 1, msh.nt);
  dgfreestream(msh, p, u0);
  u1 = 0.0;
  u.realloc(msh.ns, u0.size(1) + 1, msh.nt);
  dgconcat(u0, u1, u);
  
  // initial shock sensor
  double eps0 = 0.65;
  // At first, always add viscosity 
  sensor av(msh, d, {eps0,eps0,-4,1}, 0, u0.size(1));
  
  // Initial steps to reduce transients
  if (step0 == 0) {
    av.mksensor(u);
    dgprintf(" >>> Initial Steps <<<\n");
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 1.%d <<<\n", i);
      dgiirktime(dgnsshk, u, msh, d, p, dt/100, 1, newton);
    }
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 2.%d <<<\n", i);
      dgiirktime(dgnsshk, u, msh, d, p, dt/50, 1, newton);
    }
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 3.%d <<<\n", i);
      dgiirktime(dgnsshk, u, msh, d, p, dt/20, 1, newton);
    }
    for (int i=0; i<presteps; ++i) {
      dgprintf(" >>> Initial step 4.%d <<<\n", i);
      dgiirktime(dgnsshk, u, msh, d, p, dt/10, 1, newton);
    }
    fwritesolution(pre + "sol" + fill_int_to_string(1, 5, '0') + ".dat", u, msh);
  } else {
    // No initial steps. Load from file instead
    dgprintf(" >>> Checkpointing from %d <<<\n", step0);
    freadsolution(pre + "sol" + fill_int_to_string(step0, 5, '0') + ".dat", u, msh);
    
  }
  
  // Sensor parameters for the actual time steps
  // [0] = max viscosity = eps_0 (multiplied by h/p in the code)
  // [1] = "0" viscosity
  // [2] = log_10(average indicator) = s_0
  // [3] = log_10(half-distance) = kappa
  
  //av.pars = {.75,0,-4,1}; // explodes for p = 3
  av.pars = {.5,0,-4.5,1}; //almost works for p = 3
  //av.pars = {.5,0,-5.5,1}; // Testing for p = 4
    
  // Main loop
  int actual_steps = ( Tfinal == 0.0 ? nsteps : int(round(Tfinal / dt)) );
  for (int i=step0+1; i<=actual_steps; i++) {
    dgprintf("\n >>> Step %5d <<< \n", i);
    av.mksensor(u);
    dgiirktime(dgnsshk, u, msh, d, p, dt, order, newton);
    if (i % writeint == 0)
      fwritesolution(pre + "sol" + fill_int_to_string(i, 5, '0') + ".dat", u, msh);
  }
  
  MPI::Finalize();
}
