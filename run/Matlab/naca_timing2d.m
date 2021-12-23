% Recreates C++ code ../timing2d.cpp implemented in Matlab using subiteration solver for linear problem
function naca_timing2d()

  suffix   = "_v2";
  order    = 3;
  refine   = 12;    % amount boundary layer has been refined
  R        = 5;     % far field size
  meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
  meshname = "naca" + suffix + "_p" + num2str(order) + "_r" + num2str(refine) + "_R" + num2str(R);
  pre      = "/scratch/mfranco/2021/naca/run/results/Matlab/" + meshname + "/snaps/";
  hwing    = 0.049; % Must be value from meshes/scripts/mk_naca_v*.m
  Re       = 9.0*order/(hwing/(2^refine)); % Because h/p = 10/Re sets safe h for boundary layer, this should be safe Re
  M0       = 0.25;
  AoAdeg   = 0.0;
  dt       = 2e-4;
  Tfinal   = 0.0;   % define as 0 to use nsteps variable instead
  nsteps   = 5000;  % only used if Tfinal == 0.0
  presteps = 10;
  step0    = 1500;     % set to 1 if you have precomputed solution
  writeint = 50;
  nstages  = 3;
  maxiter  = 300;
  linerror = 1e-4;
  nmaxiter = 20;
  nlerror  = 1e-6;
  
  msh = h5freadstruct(meshdir+meshname+".h5");
  data = dginit(msh);
  fprintf("Re = %f\n", Re);

  %%%%%%%%%%%%%
  % HACK TO TEST SOLVER
  %%%%%%%%%%%%%
  jac_dir = pre+"../";
  Dij = freadarray(jac_dir+"Dij.mat");
  Oij = freadarray(jac_dir+"Oij.mat");
  
  global MatrixI; global MatrixJ;
  MatrixI = double([Dij(1,:).'; Oij(1,:).'])+1;
  MatrixJ = double([Dij(2,:).'; Oij(2,:).'])+1;
  
  % The better way to hack this but right now dgindices throws an error
  %N = size(msh.p1,1) 
  %[Dii,Djj,Oii,Ojj]=dgindices(msh,data,'drdu',N);
  %global MatrixI; global MatrixJ; 
  %MatrixI = double([Dii(:).'; Oii(:).']);
  %MatrixJ = double([Djj(:).'; Ojj(:).']);

  global bl_elems;
  bl_elems = freadarray(meshdir+meshname+"bl.mat");
  %%%%%%%%%%%%%
  % HACK TO TEST SOLVER
  %%%%%%%%%%%%%

  AoA = AoAdeg*pi/180;
  if isfinite(Re)
    wall = 2;
  else
    wall = 3;
  end
  bndcnds = [wall, 1];
  flow = [1.4, M0];
  visc = [Re, 0.72, 0.0, 1.0, 0.0]; % Re, Pr, C11 interior, C11 boundary, hscale with C11
  % TODO: [C11 interior, C11 boundary, hscale with C11] = [0, 1, 0]; Is this used by Matlab?
  vel0 = [cos(AoA), sin(AoA)];
  
  phys = physinit(msh, bndcnds, flow, visc, vel0);
  u = freestream(msh, phys);

  % Initialize solver parameters
  solver_parameters = [nlerror,nmaxiter,linerror,maxiter,0];
  % TODO: bl_elems is hardcoded into dgitprecond.cpp right now. MAKE SURE THE MESH FILE MATCHES BL.mat FILE!
  
  if step0 == 0
    % Initial steps to reduce transients
    fprintf(" >>> Initial Steps <<<\n");
    u = dgirktime(u,msh,data,phys,@dgnavierstokes, [dt/100,presteps,1], solver_parameters, 'fgmres j');
    u = dgirktime(u,msh,data,phys,@dgnavierstokes, [dt/50, presteps,1], solver_parameters, 'fgmres j');
    u = dgirktime(u,msh,data,phys,@dgnavierstokes, [dt/20, presteps,1], solver_parameters, 'fgmres j');
    u = dgirktime(u,msh,data,phys,@dgnavierstokes, [dt/10, presteps,1], solver_parameters, 'fgmres j');
    fwritearray(pre+"sol"+num2str(1, "%05d")+".dat", u);
  else
    % Don't compute initial steps. Load solution from file instead
    fprintf(" >>> Checkpointing from %d <<< \n", step0);
    u = freadarray(pre+"sol"+num2str(step0, "%05d")+".dat");
    phys.time = 0+step0*dt;
  end

  if Tfinal == 0.0
    final_step = step0+nsteps;
  else
    final_step = round(Tfinal/dt);
  end
  % Main time stepping loop
  for i = step0+1:final_step
    fprintf("\n >>> Step %5d <<< \n", i);
    u = dgirktime(u,msh,data,phys,@dgnavierstokes, [dt,1,nstages], solver_parameters, 'fgmres j');
    if mod(i, writeint) == 0
      fwritearray(pre+"sol"+num2str(i, "%05d")+".dat", u);
    end
  end
  fwritearray(pre+"sol"+num2str(final_step, "%05d")+".dat", u);
end
