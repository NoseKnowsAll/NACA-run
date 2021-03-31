#include "dg.h"
#include <ios>
#include <iostream>
#include <sstream>
#include <cmath>

const std::string suffix   = "_v2";
const int         order    = 3;
const int         refine   = 12;
const std::string meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
//const std::string meshdir  = "/scratch/mfranco/2021/naca/meshes/";
const std::string meshname = "naca" + suffix + "_p" + to_string(order) + "_r" + to_string(refine);
const std::string pre      = "/scratch/mfranco/2021/naca/run/results/" + meshname + "/snaps/";
// TODO: Technically hLE is a bit smaller than hwing, but vast majority of BL elements will be based on hwing size. Should we use hwing?
const double      hwing    = 0.049; // Make sure this is updated with correct value from naca_vX.geo
const double      Re       = 9.0*order/(hwing/(1<<refine)); // Because h/p = 10/Re sets safe h for boundary layer, this should be safe Re
const double      M0       = 0.25;
const double      AoAdeg   = 0.0;
const int         NDT      = 7;
const double      dts[NDT] = {1e-7,1e-6,1e-5,1e-4,2e-4,4e-4,1e-3}; // the dt we wish to sweep over
const int         nsteps   = 5;   // Number of iterations to compute in order to avoid timing discrepancies
const int         step0    = 5000;   // Must have a precomputed solution at this time step to begin
const int         writeint = 10;
const int         nstages  = 3;

const double      linerror = 1e-4;
const double      nlerror  = 1e-6;

void linassemble(jacarray& Ddrdu,jacarray &Odrdu, jacarray& DJ,jacarray& OJ, darray& r,
		 appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt);
std::pair<double, int> power_iteration(const jacarray& DMat,const jacarray& OMat, darray& r, darray& r2,
				       mesh& msh, data& d, double lambda0, int maxiter=500, bool verbose=false);

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
  
  // Time DG operations
  dgprintf("   dt    tassem  tpower iter lambda\n");
  for (int test = 0; test < NDT; ++test) {
    double dt = dts[test];
    double init_time = gettime();
      
    // Assemble M-dt*J
    double timer0 = gettime();
    linassemble(Ddrdu,Odrdu, DMat,OMat, r, a,msh,d,p, u,dt);
    dgprintf("%7.1e %7.3f ", dt, gettime() - timer0);
      
    // MATLAB computation of eigenvalues: Print Dmat,OMat to file
    // TODO: Determine exactly which matrix to print to file
    /* // M-dt*J
    dgprintf("Forming M-dt*J matrix\n");
    DMat = Ddrdu;
    OMat = Odrdu;
    */
    
    /* // Diagonal matrix with increasing values along the diagonal
    dgprintf("Forming example diagonal matrix\n");
    DMat = 0.0;
    OMat = 0.0;
    for (int it = 0; it < nBI; ++it) {
      for (int ic = 0; ic < DMat.size(1); ++ic) {
	DMat(ic,ic,it) = it*DMat.size(1) + ic;
      }
    }
    */
    
    /* // DMat == Mass
    dgprintf("Forming mass matrix\n");
    DMat = 0.0;
    OMat = 0.0;
    serial::bdf_add_diag_mass(DMat, OMat, dt, msh, d);
    */
    
    fwritejac(msh, d, N, DMat, OMat, pre+"../");
    return 0;

    // Compute the max eigenvalue of matrix using power iteration
    timer0 = gettime();
    k = 0.0;
    r = 1.0;
    double lambda0 = 6.0;
    auto powerit_stats = power_iteration(DMat, OMat, r, k, msh, d, lambda0, 2000, true);
    double lambda = powerit_stats.first;
    int iter = powerit_stats.second;
    dgprintf("%7.3f %4d %7.3f\n", gettime() - timer0, iter, lambda);
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

// Compute the dot product (globally) of two given solution vectors
double dotprod(const darray& x, const darray& y, mesh& msh) {
  const int N = x.size();
  double dot = std::inner_product(&x(0),&x(N),&y(0),0.0);

#ifdef ISMPI
  if (msh.is_mpi()) {
    double alldot;
    MPI_Allreduce(&dot,&alldot,1,MPI_DOUBLE,MPI_SUM,msh.COMM);
    return alldot;
  }
#endif
  return dot;
}

// Compute the 2-norm of a given x vector
double norm2(const darray& x, mesh& msh) {
  return std::sqrt(dotprod(x, x, msh));
}

// Compute the maximum eigenvalue of the matrix using the power iteration algorithm
std::pair<double, int> power_iteration(const jacarray& DMat,const jacarray& OMat, darray& r, darray& r2,
				       mesh& msh, data& d, double lambda0, int maxiter, bool verbose) {
  double lambda = 0.0;
  double r_norm = norm2(r, msh);
  double r2_norm = 0.0;
  for (int iter = 0; iter < maxiter; ++iter) {
    // Compute r2 = A*r
    matvec(msh, d, msh.porder, DMat, OMat, r, r2);
    r2_norm = norm2(r2, msh);

    // Compute approximate eigenvalue = Rayleigh quotient
    lambda = dotprod(r, r2, msh);
    lambda /= (r_norm*r_norm);
    
    if (verbose)
      dgprintf("%4d: %7.3f\n", iter, lambda);
    if (abs(lambda - lambda0) < 1e-6 * abs(lambda))
      return {lambda, iter};

    // Update variables for next iteration
    lambda0 = lambda;
    r2 /= r2_norm;
    r = r2;
    r_norm = 1.0; // r2 has already been normalized
  }
  return {lambda, maxiter};
}
