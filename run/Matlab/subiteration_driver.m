% From a set of matrices saved relative to wr_dir, named msh_name, solve Ax=b with FGMRES preconditioned with subiteration
function subiteration_driver(dt, tol, nsubiter, subtol_factor, global_precond_type, h, p)

  if nargin < 7; p = 3; end;
  if nargin < 6; h = 1e-3; end;
  if nargin < 5
    global_precond_type = "jacobi";
  else
    if ~ismember(global_precond_type, ["jacobi", "mass_inv"])
      fprintf("ERROR: global preconditioner type not one of the acceptable types!\n");
      return;
    end
  end
  if nargin < 4; subtol_factor = 1; end;
  if nargin < 3; nsubiter = 500; end;
  if nargin < 2; tol = 1e-8; end;
  if nargin < 1; dt = 1e-3; end;
  
  msh_name = sprintf("aniso_p%d_h%.0e", p, h);
  wr_dir = "/scratch/mfranco/2021/wr-les-solvers/";
  msh_dir = wr_dir+"meshes/";
  bl_file = msh_dir+msh_name+"bl.mat";
  msh_file = msh_dir+msh_name+".h5";
  results_dir = wr_dir+"results/"+msh_name+"/";
  mass_dir = results_dir+"mass/";

  fprintf("Running subiteration_driver(dt=%.1e) on mesh %s, nsubiter=%d.\n", dt, msh_name, nsubiter);

  msh = h5freadstruct(msh_file);
  global t2t;
  t2t = msh.t2t + 1; % C++ is 0-indexed
  J   = freadjac(results_dir);
  Ms  = freadarray(mass_dir+"Dv.mat");
  JD  = freadarray(results_dir+"Dv.mat");
  Dij = freadarray(results_dir+"Dij.mat");
  b   = freadarray(mass_dir+"residual.mat");
  b   = b(:);

  bl_elems = freadarray(bl_file);
  bl_elems = uint64(bl_elems(:));

  subiteration_test(J, Ms, JD, Dij, b, bl_elems, dt, tol, nsubiter, subtol_factor, global_precond_type);
  
end
