Largely research work. A sequence of tests and scripts for 3DG and implicit time stepping applied to NACA0012 airfoil. Everything below here is a note to myself about how to run these tests.

1) Create the appropriate mesh in ./meshes/  Not an exact science
   a) Modify mesh constants in naca_vX.geo (can modify from naca_orig.geo)
   b) Ensure that the boundary layer will be sufficiently resolved once it is recursively resolved
      1) Given Re, h/p = 10/Re is a reasonable size for boundary layer refinement
      2) Therefore we can resolve our mesh first and define Re based on the level of refinement: Re = 10p/h or 9p/h to be safe
   c) Run scripts/mk_naca_vX_msh.m in ml3dg with specific p to create single-threaded meshes of given geo file of order p and specified refinement level
   
2) Preprocess the mesh for run of N number of processors
   a) In ./run/, run: python preprocess.py -n N -s "suffix" -os "outputSuffix" -d "meshDirectory" -od "outputDirectory"
   
3) Run actual job to simulate flow on partitioned mesh on N processors
   a) Run needs to have an explicit dt of approximately 10^{-7} so that an implicit dt=10^{-4} will actually be competitive
   b) Tentative parameters: M0=0.25, dt_explicit=1e-7, dt_implicit=1e-4
      1) From above point 3)b)2), we should have several Re, depending on the level of refinement, p, and hwing: Re := 9p/(h*2^refine)
   c) First run for several time steps in order to reduce transients (10 steps at dt/100, dt/50, dt/20, and then dt/10)
      1) This step can be skipped once it is run once already by storing and reloading the soln vector from file with step0 = 1
   d) For implicit: record the avg matvecs, preconditioner applications, Jacobian assembles, and residual evaluations per time step
      1) Also record average wall clock time of these so that we can compare baseline timings within overall method and with explicit
      2) This should be timed only on runs we care about. What level parallelism?
   e) For explicit: residual evaluations are the only thing to note. RK4 => 1 per stage, 4 per time step
      1) Just need to measure the avg wall clock time for residual evaluations.
      2) This should be timed only on runs we care about. What level parallelism?
