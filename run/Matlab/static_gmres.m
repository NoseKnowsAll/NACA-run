% Default implementation of GMRES - not adaptive
% A can be a matrix s.t. A*x = b is the linear system to solve or
% A can be a function handle s.t. A(x) = A*x = b is linear system to solve
% precond is a function handle s.t. precond(x) = P\x for use in solving
% left-preconditioned system P\Ax = P\b
function [x, iter, residuals] = static_gmres(A, b, tol, maxiter, precond, verbose)
  
  % Initialization
  if nargin < 6
    verbose = false;
    if nargin < 5
      precond = @(x) x;
    end
  end
  
  m = length(b);
  if maxiter > m
    maxiter = m;
  end
  %b0 = precond(b); % Left preconditioning
  b0 = b;           % Right preconditioning
  residual = norm(b0);
  tol = tol*residual;
  if verbose
    fprintf("Static GMRES, reaching absolute tolerance %8.5f in %d maximum iterations.\n", tol, maxiter);
  end
  Q = b0/residual;
  omegaN = 1;
  H = [];
  residuals = [];

  for n = 1:maxiter

    % Arnoldi iteration
    if isa(A, 'function_handle')
      %vec = precond(A(Q(:,n))); % Left preconditioning
      vec = A(precond(Q(:,n))); % Right preconditioning
    else
      %vec = precond(A*Q(:,n)); % Left preconditioning
      vec = A*precond(Q(:,n)); % Right preconditioning
    end
    h = Q'*vec;
    vec = vec - Q*h;
    hN = norm(vec);
    H = [H h;
	 zeros(1, n-1) hN];

    % Givens rotation
    omegaDotH = omegaN'*h;
    denom = sqrt(omegaDotH^2 + hN^2);
    t = hN/denom;
    v = omegaDotH/denom;

    % Convergence tests
    if hN < 1e-8
      fprintf("it: %4d, converging because hN=%f is close to 0\n", n, hN);
      break;
    end
    residual = abs(t)*residual;
    % Efficient computation of residual. Equivalent to computing the following:
    e1 = [1; zeros(n,1)];
    y = H\(norm(b0)*e1);
    residual2 = norm(norm(b0)*e1 - H*y);
    x = Q*y;
    x = precond(x); % Right preconditioning
    if isa(A, 'function_handle')
      res = b - A(x);
    else
      res = b - A*x;
    end
    %residual3 = norm(precond(res)); % Left preconditioning
    residual3 = norm(res); % Right preconditioning
    if verbose
      fprintf("it: %4d, hN = %8.6f, res = %7.5f\n", n, hN, residual/norm(b0));
      fprintf("||res1|| = %f, ||res2|| = %f, ||res3|| = %f\n", residual, residual2, residual3);
    end
    residuals = [residuals residual];
    if residual < tol
      break;
    end

    % Update for next iteration
    if n < maxiter
      Q = [Q vec/hN];
      omegaN = [-t*omegaN; v];
    end
    
  end

  % Now that convergence has been met, retrieve solution x
  iter = n;
  e1 = [1; zeros(iter,1)];
  x = Q * ( H\(norm(b0)*e1) );
  x = precond(x); % Right preconditioning
  
end

