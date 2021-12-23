% Solves the A(x) = b problem using preconditioned FGMRES with subiteration preconditioner
function x = subiteration_solver(Atimes, diagA, b, bl_elems, tol, maxiter)

  precond = init_subiteration(Atimes, diagA, [], bl_elems, b, "jacobi", tol, 100, 1.0);
  restart = maxiter;
  verbose = false;
  if verbose
    fprintf("GMRES initialized successfully\n");
  end
  
  t_start = tic;
  [x, iter, residuals] = restarted_fgmres(Atimes, b, [], tol, restart, maxiter, precond, true);
  fgmres_timer = toc(t_start);

  if verbose
    fprintf("total iterations: %d\n", iter);
    fprintf("time: %6.2f\n", fgmres_timer);
  end
  
end
