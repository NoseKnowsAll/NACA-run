% Perform a linear iterative method to solve Ax=b to a given relative tolerance using initial guess x0.
% iteration is a function handle to perform one iteration of the method of the form x_n = iteration(x_{n-1})
function [x, iter, residuals] = iterative_method(A, b, x0, tol, maxiter, iteration, verbose)
  
  % Initialization
  if isempty(x0)
    x = zeros(size(b));
  else
    x = x0;
  end
  beta = norm(b);
  residuals = [];

  if verbose
    fprintf("Linear iterative method ");
    if ~isempty(tol)
      fprintf("reaching relative tolerance %8.2e ", tol);
    end
    fprintf("in %d maximum iterations.\n", maxiter);
  end

  % Iterations
  for it = 1:maxiter
    x = iteration(x);
    % Convergence tests
    if ~isempty(tol)
      if isa(A, 'function_handle')
	residuals(it) = norm(b - A(x));
      else
	residuals(it) = norm(b - A*x);
      end
      %if verbose
	%fprintf("%4d: ||res|| = %8.2e\n", it, residuals(it));
      %end
      if residuals(it) < tol*beta
	if verbose
	  fprintf("Iterative solver converging after %d iterations because %8.2e < %8.2e\n", it, residuals(it), tol*beta);
	end
	iter = it;
	return;
      end
    end
  end

  iter = maxiter;
  if verbose
    fprintf("Iterative method completed all %d iterations.\n", maxiter);
  end
  
end
