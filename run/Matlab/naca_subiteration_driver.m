% Solves the linear Ax=b problem from the NACA airfoil N-S solver
function naca_subiteration_driver(dt, tol, nsubiter, global_precond_type)

  if nargin < 4
    global_precond_type = "jacobi";
    if nargin < 3
      nsubiter = 500;
      if nargin < 2
	tol = 1e-8;
	if nargin < 1
	  dt = 1e-3;
	end
      end
    end
  else
    if ~ismember(global_precond_type, ["jacobi", "mass_inv"])
      fprintf("ERROR: global preconditioner type not one of the acceptable types!\n");
      return;
    end
  end

  msh_name = "naca_v2_p3_r12";
  bl_file = "/scratch/mfranco/2021/naca/meshes/"+msh_name+"bl.mat";
  msh_file = "/scratch/mfranco/2021/naca/run/partitioned/"+msh_name+".h5";
  results_dir = "/scratch/mfranco/2021/naca/run/results/"+msh_name+"/";
  mass_dir = results_dir+"mass/";

  fprintf("Running subiteration_driver(dt=%.1e) on mesh %s, nsubiter=%d.\n", dt, msh_name, nsubiter);

  J   = freadjac(results_dir);
  Ms  = freadarray(mass_dir+"Dv.mat");
  JD  = freadarray(results_dir+"Dv.mat");
  Dij = freadarray(results_dir+"Dij.mat");
  b   = freadarray(mass_dir+"residual.mat");
  b   = b(:);

  bl_elems = freadarray(bl_file);
  bl_elems = uint64(bl_elems(:));

  subiteration_test(J, Ms, JD, Dij, b, bl_elems, dt, tol, nsubiter, global_precond_type);
  
end
