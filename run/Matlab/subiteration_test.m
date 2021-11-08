% Now that information is initialized, actually test FGMRES preconditioned by subiterations
function subiteration_test(J, Ms, JD, Dij, b, bl_elems, dt, tol, nsubiter, subtol_factor, global_precond_type, divide_by_dt)

  if ~exist('divide_by_dt','var')
    divide_by_dt = false; % Assume working with 3DG matrices
  end

  if divide_by_dt
    % For testing MFEM: should be mfem_time_dependent_jacobian
    Atimes = @(x)mfem_time_dependent_jacobian(J, Ms, dt, x);
    diagA = Ms/dt - JD;
  else
    % For testing 3DG: do not divide by dt
    Atimes = @(x)time_dependent_jacobian(J, Ms, dt, x);
    diagA = Ms-dt*JD;
  end

  %fprintf("Working on condition number...\n");
  %sparseA = construct_sparse_matrix(diagA, Dij);
  %disp(cond(full(sparseA)));
  %return;

  precond = init_subiteration(Atimes, diagA, Ms, bl_elems, b, global_precond_type, tol, nsubiter, subtol_factor);
  %precond = init_jacobi(diagA);
  %precond = init_mass_inv(Ms);
  %precond = @(x) x;

  maxiter = 500;
  restart = 500;
  fprintf("GMRES initialized successfully\n");
  t_start = tic;

  %[x, iter, residuals] = static_gmres(Atimes, b, [], tol, maxiter, precond, "flexible", true);
  [x, iter, residuals] = restarted_fgmres(Atimes, b, [], tol, restart, maxiter, precond, true);

  global fgmres_timer;
  fgmres_timer = toc(t_start);
  fprintf("total iterations: %d\n", iter);
  global inner_iterations;
  if inner_iterations > 0
    fprintf("inner iterations: %d\n", inner_iterations);
  end
  fprintf("time: %6.2f\n", fgmres_timer);
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
