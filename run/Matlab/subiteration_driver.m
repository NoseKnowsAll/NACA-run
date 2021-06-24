function subiteration_driver(dt, nsubiter, global_precond_type, h, p)

  if nargin < 5
    p = 3;
    if nargin < 4
      h = 1e-3;
      if nargin < 3
	global_precond_type = "jacobi";
	if nargin < 2
	  nsubiter = 500;
	  if nargin < 1
	    dt = 1e-3;
	  end
	end
      else
	if ~ismember(global_precond_type, ["jacobi", "mass_inv"])
	  fprintf("ERROR: global preconditioner type not one of the acceptable types!\n");
	  return;
	end
      end
    end
  end
  
  msh_name = sprintf("aniso_p%d_h%.0e", p, h);
  wr_dir = "/scratch/mfranco/2021/wr-les-solvers/";
  msh_dir = wr_dir+"meshes/";
  bl_file = msh_dir+msh_name+"bl.mat";
  msh_file = msh_dir+msh_name+".h5";
  results_dir = wr_dir+"results/"+msh_name+"/";
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

  Atimes  = @(x)time_dependent_jacobian(J, Ms, dt, x);
  diagA = Ms-dt*JD;

  %fprintf("Working on condition number...\n");
  %sparseA = construct_sparse_matrix(diagA, Dij);
  %disp(cond(full(sparseA)));
  %return;

  precond = init_subiteration(Atimes, diagA, Ms, bl_elems, b, global_precond_type, nsubiter);
  %precond = init_jacobi(Atimes, diagA, b);
  %precond = init_mass_inv(Ms);
  %precond = @(x) x;

  maxiter = 500;
  restart = 500;
  tol = 1e-8;
  fprintf("GMRES initialized successfully\n");
  t_start = tic;

  [x, iter, residuals] = static_gmres(Atimes, b, [], tol, maxiter, precond, "flexible", true);
  
  t_gmres = toc(t_start);
  fprintf("total iterations: %d\n", iter);
  fprintf("time: %6.2f\n", t_gmres);
  fprintf("||Ax-b|| = %f\n", norm(Atimes(x)-b));
  
end

% Explicitly construct the block-diagonal sparse matrix A
% defined by diagA values on Dij indices (i,j)
function A = construct_sparse_matrix(diagA, Dij)
  ni = size(diagA,1)*size(diagA,3);
  Ai = double(Dij(1,:).')+1; % MATLAB is 1-indexed
  Aj = double(Dij(2,:).')+1; % MATLAB is 1-indexed
  Av = [diagA(:)];
  A = sparse(Ai, Aj, Av, ni, ni);
end

% Test that matvec done in Matlab is same as in C++
function test_matvec(Atimes, b, mass_dir)
  Ab1 = Atimes(b);
  Ab2 = freadarray(mass_dir+"benchmark_matvec.mat");
  Ab2 = Ab2(:);

  norm(b)
  norm(Ab1)
  norm(Ab2)
  fprintf("diff = %f\n", norm(Ab1-Ab2));
end
