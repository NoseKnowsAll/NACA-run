#include "dg.h"
#include <ios>
#include <iostream>
#include <sstream>

const std::string suffix   = "_v2";
const int         order    = 3;
const int         refine   = 12;
const std::string meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
//const std::string meshdir  = "/scratch/mfranco/2021/naca/meshes/";
const std::string meshname = "naca" + suffix + "_p" + to_string(order) + "_r" + to_string(refine);
const std::string bl_file  = meshdir + meshname + "bl.mat";
const std::string pre      = "/scratch/mfranco/2021/naca/run/results/" + meshname + "/snaps/";
// TODO: Technically hLE is a bit smaller than hwing, but vast majority of BL elements will be based on hwing size. Should we use hwing?
const double      hwing    = 0.049; // Make sure this is updated with correct value from naca_vX.geo
const double      Re       = 9.0*order/(hwing/(1<<refine)); // Because h/p = 10/Re sets safe h for boundary layer, this should be safe Re
const double      M0       = 0.25;
const double      AoAdeg   = 0.0;
const int         NDT      = 7;
const double      dts[NDT] = {1e-7,1e-6,1e-5,1e-4,2e-4,4e-4,1e-3}; // the dt we wish to sweep over
const int         nsteps   = 2;    // Number of iterations to compute in order to avoid timing discrepancies
const int         step0    = 5000; // Must have a precomputed solution at this time step to begin
const int         writeint = 10;
const int         nstages  = 3;

const double      linerror = 1e-4;
const double      nlerror  = 1e-6;

void linassemble(jacarray& Ddrdu,jacarray &Odrdu, jacarray& DJ,jacarray& OJ, darray& r,
		 appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt);

void zero_nonbl(jacarray& Ddrdu, jacarray &Odrdu, mesh& msh, const iarray& bl_elems);

double gettime() {
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if (np > 1)
    return MPI_Wtime();
  else
    return threedg::timer();
}

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
  int maxiter = 2000;
  int restart = 200;
  // j => Jacobi, i => ILU, d => direct, b => boundary layer. See: dgitprecond.h
  auto linsolver = LinearSolverOptions::gmres("m", linerror, maxiter, restart);
  auto newton = NewtonOptions(linsolver, nlerror);
  
  // Set up the solution variable
  darray u;
  dgfreestream(msh, p, u);
  // Allocate space for internal arrays
  int N = u.size(1);
  int nBI  = msh.nBI();  // == nt if np==1
  int nBIN = msh.nBIN(); // == nt if np==1
  darray r(d.ns, N, nBIN);
  darray k(d.ns, N, nBIN);
  jacarray Ddrdu(N*d.ns,  N*d.ns, nBI);
  jacarray Odrdu(N*d.nes, N*d.ns, msh.nf, nBI);
  jacarray DJ   (N*d.ns,  N*d.ns, nBI);
  jacarray OJ   (N*d.nes, N*d.ns, msh.nf, nBI);
  darray results(2, NDT);

  // Initialize boundary layer elements from file
  iarray bl_elems;
  FILE *fid = fopen(bl_file.c_str(), "rb");
  freadarray(fid, bl_elems);
  for (int i = 0; i < bl_elems.size(); ++i)
    bl_elems(i) -= 1; // MATLAB is 1-indexed...
  fclose(fid);
  
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
      double init_time = gettime();
      
      // Assemble M-dt*J
      double timer0 = gettime();
      linassemble(Ddrdu,Odrdu, DJ,OJ, r, a,msh,d,p, u,dt);
      // Zero-out non-boundary layer elements
      zero_nonbl(Ddrdu,Odrdu, msh, bl_elems);
      //fwritejac(msh, d, N, Ddrdu, Odrdu, "./debug_data/");
      //return 0;
      dgprintf("%7.3f ", gettime() - timer0);
      
      // Linear solve (M-dt*J)k = r
      k = 0.0;
      timer0 = gettime();
      /*
      //  First initialize preconditioner - Direct solver for only diagonal portion of M-dt*J
      DJ = Ddrdu;
      OJ = 0.0;
      dgitprecond precond(msh, d, DJ, OJ, "j", false);
      */
      //  Then solve the linear system
      std::pair<double,int> linsolve_stats = linsolve(msh, d, Ddrdu, Odrdu, r, k, newton.linsolver);
      //std::pair<double,int> linsolve_stats = dgkrylovsolve(msh, d, Ddrdu, Odrdu, r, k, newton.linsolver, &precond);
      dgprintf("%7.3f ", gettime() - timer0);
      if (newton.linsolver.solver != "direct") {
	dgprintf("%3d", linsolve_stats.second);
	iter = linsolve_stats.second;
      }
      dgprintf("\n");
      
      double time = gettime() - init_time;
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

// Assemble Ddrdu,Odrdu to contain M-dt*J information. Assemble J and r as well
void linassemble(jacarray& Ddrdu,jacarray &Odrdu, jacarray& DJ,jacarray &OJ, darray& r,
		 appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt) {
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if (np > 1) {
    pararray pr(msh, r);
    mpi::parassembleB(a,u,r, Ddrdu, Odrdu, msh, d, p);
    pr.communicate();
    mpi::parassembleI(a,u,r, Ddrdu, Odrdu, msh, d, p);
    DJ = Ddrdu;
    OJ = Odrdu;
    
    mpi::bdf_add_diag_mass(Ddrdu, Odrdu, dt, msh, d);
    pr.waitforall();
    
  } else {
    dgassemble(a, u, r, Ddrdu, Odrdu, msh, d, p);
    DJ = Ddrdu;
    OJ = Odrdu;
    
    serial::bdf_add_diag_mass(Ddrdu, Odrdu, dt, msh, d);
  }
}

// In-place zero-out non-boundary layer element rows of Jacobian matrix
void zero_nonbl(jacarray& Ddrdu, jacarray &Odrdu, mesh& msh, const iarray& bl_elems) {
  int nt = msh.nt;
  int nf = msh.nf;
  int nBI = msh.nBI();
  
  // Initialize local non-boundary layer elements
  iarray local_non_bls(nt-bl_elems.size());
  iarray local_bls(bl_elems.size());
  int curr_i = 0;
  int curr_bl_i = 0;
  for (int it = 0; it < nBI; ++it) {
    int git = msh.getitglobal(it);
    bool found_in_bl = false;
    for (int i = 0; i < bl_elems.size(); ++i) {
      if (bl_elems(i) == git) {
	found_in_bl = true;
	break;
      }
    }
    if (!found_in_bl) {
      if (curr_i == local_non_bls.size(0))
	dgerror("SETUP NON-BOUNDARY LAYER ELEMENTS INCORRECTLY!");
      local_non_bls(curr_i) = it;
      curr_i++;
    } else {
      if (curr_bl_i == local_bls.size(0))
	dgerror("SETUP BOUNDARY LAYER ELEMENTS INCORRECTLY!");
      local_bls(curr_bl_i) = it;
      curr_bl_i++;
    }
  }
  local_non_bls.resize(curr_i);
  local_bls.resize(curr_bl_i);

  // Zero out non-boundary layer elements
  for (int i = 0; i < local_non_bls.size(0); ++i) {
    int it = local_non_bls(i);
    for (int i1 = 0; i1 < Ddrdu.size(1); ++i1) {
      for (int i0 = 0; i0 < Ddrdu.size(0); ++i0) {
	Ddrdu(i0,i1,it) = 0.0;
      }
    }
    for (int i2 = 0; i2 < Odrdu.size(2); ++i2) {
      for (int i1 = 0; i1 < Odrdu.size(1); ++i1) {
	for (int i0 = 0; i0 < Odrdu.size(0); ++i0) {
	  Odrdu(i0,i1,i2,it) = 0.0;
	}
      }
    }
  }
  // Have to also zero-out connections with neighbors outside of boundary layer
  for (int i = 0; i < local_bls.size(0); ++i) {
    int it = local_bls(i);
    for (int j = 0; j < nf; ++j) {
      int it2 = msh.t2t(j,it);
      if (it2>=0) {
	int git2 = msh.getitglobal(it2);
	bool found_in_bl = false;
	for (int i2 = 0; i2 < bl_elems.size(); ++i2) {
	  if (bl_elems(i2) == git2) {
	    found_in_bl = true;
	    break;
	  }
	}
	if (!found_in_bl) {
	  for (int i1 = 0; i1 < Odrdu.size(1); ++i1) {
	    for (int i0 = 0; i0 < Odrdu.size(0); ++i0) {
	      Odrdu(i0,i1,j,it) = 0.0;
	    }
	  }
	}
      }
    }
  }
  
}
