% Now that information is initialized, actually test FGMRES preconditioned by subiterations
function subiteration_test(J, Ms, JD, Dij, b, bl_elems, dt, tol, nsubiter, subtol_factor, global_pre, inner_pre, local_divide_by_dt)

  if ~exist('divide_by_dt','var')
    divide_by_dt = false; % Assume working with 3DG matrices
  end

  if divide_by_dt
    % For testing MFEM: should be mfem_time_dependent_jacobian
    if global_pre == "ilu" || inner_pre == "ilu"
      Atimes = mfem_assemble_time_dependent_jacobian(J, Ms, dt);
    else
      Atimes = @(x)mfem_time_dependent_jacobian(J, Ms, dt, x);
    end
    diagA = Ms/dt - JD;
  else
    % For testing 3DG: do not divide by dt
    if global_pre == "ilu" || inner_pre == "ilu"
      Atimes = assemble_time_dependent_jacobian(J, Ms, dt);
    else
      Atimes = @(x)time_dependent_jacobian(J, Ms, dt, x);
    end
    diagA = Ms-dt*JD;
  end

  t_start = tic;
  precond = init_subiteration(Atimes, diagA, Ms, bl_elems, b, global_pre, inner_pre, tol, nsubiter, subtol_factor);
  ass_timer = toc(t_start);
  fprintf("GMRES initialization time: %6.2f\n", ass_timer);
  
  maxiter = 500;
  restart = 500;
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
  fprintf("solve time: %6.2f\n", fgmres_timer);
  if isa(Atimes, 'function_handle')
    fprintf("||Ax-b|| = %f\n", norm(Atimes(x)-b));
  else
    fprintf("||Ax-b|| = %f\n", norm(Atimes*x-b));
  end
  
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
