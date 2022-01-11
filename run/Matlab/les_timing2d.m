% Recreates Per's C++ code /scratch/lesflow_src/les2d.cpp implemented in Matlab using subiteration solver for linear problem
function les_timing2d()

  order    = 3;
  meshdir  = "/scratch/mfranco/2021/naca/run/partitioned/";
  meshname = "les" + "_p" + num2str(order);
  pre      = "/scratch/mfranco/2021/naca/run/results/Matlab/" + meshname + "/snaps/";
  Re       = 60e3;
  M0       = 0.10;
  AoAdeg   = 30.0;
  dt       = 1e-3;
  Tfinal   = 0.0;   % define as 0 to use nsteps variable instead
  nsteps   = 10;    % only used if Tfinal == 0.0
  step0    = 270;   % Final computed step from C++ code is 270: t=2.7
  writeint = 10;
  nstages  = 2;
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
  %[Dii,Djj,Oii,Ojj]=dgindices(msh,data,'drducdg',N);
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
  vel0 = [cos(AoA), sin(AoA)];
  Pr = dot(vel0,vel0)/flow(1)/(1.0^(flow(1)-1))/(M0*M0); % Internally computed by C++ physinit::nsisentrop
  visc = [Re, Pr, 0.0, 1.0, 0.0]; % Re, Pr, C11 interior, C11 boundary, hscale with C11
  % TODO: [C11 interior, C11 boundary, hscale with C11] = [0, 1, 0]; Is this used by Matlab?
  
  phys = physinit(msh, bndcnds, flow, visc, vel0);
  phys.uinf = phys.uinf(1:end-1) % Energy is not computed in dgnsisentrop
  u = freestream(msh, phys);

  % Initialize solver parameters
  solver_parameters = [nlerror,nmaxiter,linerror,maxiter,0];
  % TODO: bl_elems is hardcoded into dgitprecond.cpp right now. MAKE SURE THE MESH FILE MATCHES BL.mat FILE!
  
  %if step0 == 0
  %  % Initial steps to reduce transients
  %  fprintf(" >>> Initial Steps <<<\n");
  %  u = dgirktime(u,msh,data,phys,@dgnsisentrop, [dt/100,presteps,1], solver_parameters, 'fgmres j');
  %  u = dgirktime(u,msh,data,phys,@dgnsisentrop, [dt/50, presteps,1], solver_parameters, 'fgmres j');
  %  u = dgirktime(u,msh,data,phys,@dgnsisentrop, [dt/20, presteps,1], solver_parameters, 'fgmres j');
  %  u = dgirktime(u,msh,data,phys,@dgnsisentrop, [dt/10, presteps,1], solver_parameters, 'fgmres j');
  %  fwritearray(pre+"sol"+num2str(1, "%05d")+".dat", u);
  %else
    % Don't compute initial steps. Load solution from file instead
    fprintf(" >>> Checkpointing from %d <<< \n", step0);
    u = freadarray(pre+"sol"+num2str(step0, "%05d")+".dat");
    phys.time = 0+step0*dt;
  %end

  if Tfinal == 0.0
    final_step = step0+nsteps;
  else
    final_step = round(Tfinal/dt);
  end
  % Main time stepping loop
  for i = step0+1:final_step
    fprintf("\n >>> Step %5d <<< \n", i);
    u = dgirktime(u,msh,data,phys,@dgnsisentrop, [dt,1,nstages], solver_parameters, 'fgmres s');
    if mod(i, writeint) == 0
      fwritearray(pre+"sol"+num2str(i, "%05d")+".dat", u);
    end
  end
  fwritearray(pre+"sol"+num2str(final_step, "%05d")+".dat", u);
end
