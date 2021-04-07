% Restarted GMRES 
function [x, iter, residuals] = restarted_gmres(A, b, tol, maxiter, restart, adaptive)
  % initialization
  if nargin < 6
    adaptive = false;
  end
  m = length(b);
  actual_tol = tol*(1+norm(b));
  res = b;
  x = zero(m,1);
  residuals = [];

  % Outer GMRES restarting loop
  for outer_it = 1:restart:maxiter
    % Runs GMRES for at most restart iterations
    if adaptive
      [dx, inner_it, inner_res] = adaptive_gmres(A, res, tol, min([restart maxiter-outer_it]));
    else
      [dx, inner_it, inner_res] = static_gmres(A, res, tol, min([restart maxiter-outer_it]));
    end
    residuals = [residuals inner_res];
    
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
