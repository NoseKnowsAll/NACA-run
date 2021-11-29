#include "dg.h"
#include <ios>
#include <iostream>
#include <sstream>
#include <cmath>

const std::string suffix   = "_v2";
const int         order    = 3;
const int         refine   = 12; // amount boundary layer has been refined
const int         R        = 40; // far field size
const std::string meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
//const std::string meshdir  = "/scratch/mfranco/2021/naca/meshes/";
const std::string meshname = "naca" + suffix + "_p" + to_string(order) + "_r" + to_string(refine) + "_R" + to_string(R);
const std::string pre      = "/scratch/mfranco/2021/naca/run/results/" + meshname + "/snaps/";
// TODO: Technically hLE is a bit smaller than hwing, but vast majority of BL elements will be based on hwing size. Should we use hwing?
const double      hwing    = 0.049; // Make sure this is updated with correct value from naca_vX.geo
const double      Re       = 9.0*order/(hwing/(1<<refine)); // Because h/p = 10/Re sets safe h for boundary layer, this should be safe Re
const double      M0       = 0.25;
const double      AoAdeg   = 0.0;
const double      dt       = 1e-3;
const int         step0    = 5000;   // Must have a precomputed solution at this time step to begin

void linassemble(jacarray& Ddrdu,jacarray &Odrdu, jacarray& DJ,jacarray& OJ, darray& r,
		 appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt);

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
  
  // Set up the solution variable
  darray u;
  dgfreestream(msh, p, u);
  // Allocate space for internal arrays
  int N = u.size(1);
  int nBI  = msh.nBI();  // == nt if np==1
  int nBIN = msh.nBIN(); // == nt if np==1
  darray r(d.ns, N, nBI);
  darray k(d.ns, N, nBI);
  jacarray Ddrdu(N*d.ns,  N*d.ns, nBI);
  jacarray Odrdu(N*d.nes, N*d.ns, msh.nf, nBIN);
  jacarray DMat (N*d.ns,  N*d.ns, nBI);
  jacarray OMat (N*d.nes, N*d.ns, msh.nf, nBIN);
  
  // Don't compute initial steps. Load soln from file instead
  dgprintf(" >>> Checkpointing from %d <<<\n", step0);
  freadsolution(pre + "sol" + fill_int_to_string(step0, 5, '0') + ".dat", u, msh);
  
  // Assemble M-dt*J, J
  double timer0 = gettime();
  linassemble(Ddrdu,Odrdu, DMat,OMat, r, a,msh,d,p, u,dt);
  // Print J to file
  fwritejac(msh, d, N, DMat,OMat, pre+"../");
  fwritearray(pre+"../mass/residual.mat", r);
    
  // DMat == Mass
  dgprintf("Forming mass matrix\n");
  DMat = 0.0;
  OMat = 0.0;
  serial::bdf_add_diag_mass(DMat, OMat, dt, msh, d);
  // Print M to file
  fwritejac(msh, d, N, DMat, OMat, pre+"../mass/");
  
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
