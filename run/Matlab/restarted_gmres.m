% Restarted GMRES 
function [x, iter] = restarted_gmres(A, b, tol, p, maxiter, adaptive)
  % initialization
  if nargin < 6
    adaptive = false;
  end
  m = length(b);
  actual_tol = tol*(1+norm(b));
  res = b;
  x = zero(m,1);

  % Outer GMRES restarting loop
  for outer_it = 1:p:maxiter
    % Runs GMRES for at most p iterations
    if adaptive
      [dx, inner_it] = adaptive_gmres(A, res, tol, min([p maxiter-outer_it]));
    else
      [dx, inner_it] = static_gmres(A, res, tol, min([p maxiter-outer_it]));
    end
    
    x = x + dx;
    if isa(A, 'function_handle')
      res = b - A(x);
    else
      res = b - A*x;
    end
    
    if (norm(res) < actual_tol)
      break;
    end
  end

  iter = inner_it + outer_it;
  
end
