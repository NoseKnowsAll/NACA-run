% Now that information is initialized, actually test FGMRES preconditioned by subiterations
function subiteration_test(J, Ms, JD, Dij, b, bl_elems, dt, tol, nsubiter, global_precond_type)

  Atimes = @(x)mfem_time_dependent_jacobian(J, Ms, dt, x);
  diagA = Ms/dt-JD;

  %fprintf("Working on condition number...\n");
  %sparseA = construct_sparse_matrix(diagA, Dij);
  %disp(cond(full(sparseA)));
  %return;

  precond = init_subiteration(Atimes, diagA, Ms, bl_elems, b, global_precond_type, tol, nsubiter);
  %precond = init_jacobi(diagA);
  %precond = init_mass_inv(Ms);
  %precond = @(x) x;

  maxiter = 500;
  restart = 500;
  fprintf("GMRES initialized successfully\n");
  t_start = tic;

  %[x, iter, residuals] = static_gmres(Atimes, b, [], tol, maxiter, precond, "flexible", true);
  [x, iter, residuals] = restarted_fgmres(Atimes, b, [], tol, restart, maxiter, precond, true);
  
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
