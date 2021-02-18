#include "dg.h"
#include "dgavtools.h"
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
const int         NDT      = 7;
const double      dts[NDT] = {1e-7,1e-6,1e-5,1e-4,2e-4,4e-4,1e-3}; // the dt we wish to sweep over
const int         nsteps   = 5;   // Number of iterations to compute in order to avoid timing discrepancies
const int         step0    = 1;   // Must have a precomputed solution at this time step to begin
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

void linassemble(jacarray& Ddrdu,jacarray &Odrdu, darray& r, appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt);

int main(int argc, char **argv) {
  
  MPI_Init(&argc, &argv);

  // Initialize data and physics of N-S simulation according to our parameters
  mesh msh;
  data d;
  phys p;
  appl a = dgnavierstokes;
  
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
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
  auto linsolver = LinearSolverOptions::gmres("jj0dpjj", linerror, maxiter, restart); // j => Jacobi, i => ILU, d => direct
  auto newton = NewtonOptions(linsolver, nlerror);
  
  // Set up the solution variable
  darray u;
  dgfreestream(msh, p, u);
  // Allocate space for internal arrays
  int N = u.size(1);
  int nBI = msh.nBI(); // == nt if np==1
  darray r(d.ns, N, nBI);
  darray k(d.ns, N, nBI);
  jacarray Ddrdu(N*d.ns, N*d.ns, nBI);
  jacarray Odrdu(N*d.nes, N*d.ns, msh.nf, nBI);
  darray results(2, NDT);
  
  // Don't compute initial steps. Load soln from file instead
  dgprintf(" >>> Checkpointing from %d <<<\n", step0);
  freadsolution(pre + "sol" + fill_int_to_string(step0, 5, '0') + ".dat", u, msh);
  
  // Time DG operations
  dgprintf(" tassem  tsolve iter\n");
  for (int test = 0; test < NDT; ++test) {
    double dt = dts[test];
    double min_time = 1.0/0.0;
    int iter = 0;
    for (int i = 0; i < nsteps; i++) {
      set_timer();
      // Assemble M-dt*J
      double timer0 = threedg::timer();
      linassemble(Ddrdu,Odrdu, r, a,msh,d,p, u,dt);
      dgprintf("%7.3f ", threedg::timer() - timer0);
      // Linear solve (M-dt*J)k = r
      k = 0.0;
      timer0 = threedg::timer();
      std::pair<double,int> linsolve_stats = linsolve(msh, d, Ddrdu, Odrdu, r, k, newton.linsolver);
      dgprintf("%7.3f ", threedg::timer() - timer0);
      if (newton.linsolver.solver != "direct") {
	dgprintf("%3d", linsolve_stats.second);
	iter = linsolve_stats.second;
      }
      dgprintf("\n");
      
      double time = get_timer();
      if (time < min_time)
	min_time = time;
    }
    results(0,test) = iter;
    results(1,test) = min_time;
  }
  // speedup: amount faster than dt=1e-4 it would be to reach t=1.0 (IGNORING ASSEMBLY), >1 => more efficient choice for dt
  dgprintf("   dt   iter assem+solve speedup\n");
  for (int test = 0; test < NDT; ++test) {
    int comparison_index = 3;
    double speedup = dts[test]/results(0,test)*results(0,comparison_index)/dts[comparison_index];
    dgprintf("%7.1e %3d %10.3f %7.3f\n", dts[test], static_cast<int>(results(0,test)), results(1,test), speedup);
  }
  
  MPI_Finalize();
}

// Assemble Ddrdu,Odrdu to contain M-dt*J information. Assemble r as well
void linassemble(jacarray& Ddrdu,jacarray &Odrdu, darray& r, appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt) {
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &np);
  if (np > 1) {
    mpi::parassembleB(a,u,r, Ddrdu, Odrdu, msh, d, p);
    mpi::parassembleI(a,u,r, Ddrdu, Odrdu, msh, d, p);
    mpi::bdf_add_diag_mass(Ddrdu, Odrdu, dt, msh, d);
  } else {
    dgassemble(a, u, r, Ddrdu, Odrdu, msh, d, p);
    serial::bdf_add_diag_mass(Ddrdu, Odrdu, dt, msh, d);
  }
}


