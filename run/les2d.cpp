#include "dg.h"
#include "dgavtools.h"
#include "dgsolvers_mpi.h"
#include "dgkoornwinder.h"
#include "mathutil.h"
#include "fortwrap.h"

const bool       INIT_MATLAB = true;
const int              order = 3;
const std::string meshdir    = "/scratch/mfranco/2021/naca/run/partitioned/";
const std::string meshname   = "les_p" + to_string(order);
const std::string pre        = "/scratch/mfranco/2021/naca/run/results/Matlab/" + meshname + "/snaps/";
const double             Re  = 60e3;
const double             M0  = 0.10;

const double          C11bnd = 1e0;
const double          C11int = 0;
const double              dt = 0.01;
const double          Tfinal = 15.0;
const int             nsteps = int(round(Tfinal / dt));
const int              step0 = 270;

void linassemble(jacarray& Ddrdu,jacarray &Odrdu, jacarray& DJ,jacarray& OJ, darray& r,
		 appl& a,mesh& msh,data& d,phys& p, const darray& u,double dt);

int main(int argc, char **argv) {
  MPI::Init(argc, argv);

  mesh msh;
  data d;
  phys p;
  appl a = dgnsisentrop;

  int np = MPI::COMM_WORLD.Get_size();
  msh.readfile_mpi(meshdir+meshname, np);

  dginit(msh, d);

  using dg::physinit::FarFieldQty;
  dg::physinit::nsisentrop(
      msh,
      {2, 1}, // bndcnds
      {Re, 0.72, C11int, C11bnd, 0.0},  // pars
      1.0,                       // far field density
      M0,                        // far field Mach
      {sqrt(3)/2, 0.5},                // far field velocity
      &p);

  darray u;
  dgfreestream(msh, p, u);
  
  if (INIT_MATLAB) { // initMatlab.cpp
    dgprintf(" Initializing for Matlab\n");
    
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
    
  }
  else { // timing2d.cpp
    dgprintf(" Running simulation in C++\n");
    
    auto linsolver = LinearSolverOptions::gmres("j", 1e-4, 100, 100);
    auto newton = NewtonOptions(linsolver, 1e-6);

    if (step0 == 0) {

      mpi::dgirktime(dgnsisentrop, u, msh, d, p, dt/10, 3, dirk_coeffs(2), newton, "");
      fwritesolution(pre + "sol2D" + fill_int_to_string(0, 5, '0') + ".dat", u, msh);
      p.time = 0.0;
    }
    else {
      freadsolution(pre + "sol2D" + fill_int_to_string(step0, 5, '0') + ".dat", u, msh);
      p.time = step0*dt;
    }

    // Main loop
    for (int i=step0+1; i<=nsteps; i++) {
      dgprintf("\n >>> Step %5d <<< \n", i);
      //dgrktime(dgnsisentrop, u, msh, d, p, dt/tsub, tsub);
      mpi::dgirktime(dgnsisentrop, u, msh, d, p, dt/2, 2, dirk_coeffs(2), newton, "");
      fwritesolution(pre + "sol2D" + fill_int_to_string(i, 5, '0') + ".dat", u, msh);
    }
  }
  
  MPI::Finalize();
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
