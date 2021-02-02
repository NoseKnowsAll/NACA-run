1) Create the appropriate mesh in ./meshes/  Not an exact science
   a) Modify mesh constants in naca_vX.geo (can modify from naca_orig.geo)
   b) Ensure that the boundary layer will be sufficiently resolved once it is recursively resolved
      1) Given Re, h/p = 10/Re is a reasonable size for boundary layer refinement
      2) Therefore we can resolve our mesh first and define Re based on the level of refinement: Re = 10p/h or 9p/h to be safe
   c) Run scripts/mk_naca_vX_msh.m in ml3dg with specific p to create a single-threaded mesh of given geo file of order p
   
2) Preprocess the mesh for run of N number of processors
   a) Copy over final created single-threaded mesh from ./meshes/ to ./run/partitioned/
   b) In ./run/, run: python preprocess.py -n N -s "suffix" -ns "newSuffix" -d "directoryToFindMesh"
   
3) Run actual job to simulate flow on partitioned mesh on N processors
   a) Run needs to have an explicit dt of approximately 10^{-7} so that an implicit dt=10^{-4} will actuall be competitive
   b) Tentative parameters: Re=3e6, M0=0.25, dt_explicit=1e-7, dt_implicit=1e-4
   c) First run for several time steps in order to reduce transients (10 steps at dt/100, dt/50, dt/20, and then dt/10)
   d) For implicit: record the avg matvecs, preconditioner applications, Jacobian assembles, and residual evaluations per time step
      1) Also record average wall clock time of these so that we can compare baseline timings within overall method and with explicit
   e) For explicit: residual evaluations are the only thing to note. RK4 => 1 per stage, 4 per time step
      1) Just need to measure the avg wall clock time for residual evaluations
