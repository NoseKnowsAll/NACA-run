% Default implementation of GMRES - not adaptive
% A can be a matrix s.t. A*x = b is the linear system to solve or
% A can be a function handle s.t. A(x) = A*x = b is linear system to solve
% precond is a function handle s.t. precond(x) = P\x for use in solving
% (a) left-preconditioned system P\Au = P\b
% or (b) right-preconditioned system AP\u = b, u=Px
% Can specify preconditioning of GMRES method to be "left", "right", or "flexible".
% Default method: "flexible" => FGMRES.
function [x, iter, residuals] = static_gmres(A, b, tol, maxiter, precond, method, verbose)
  
  % Initialization
  if nargin < 7
    verbose = false;
    if nargin < 6
      method = "flexible"
      if nargin < 5
	precond = @(x) x;
      end
    else
      if ~(ismember(method, ["left", "right", "flexible"]))
	fprintf("Invalid method specified for GMRES. Reseting to FGMRES...\n");
	method = "flexible";
      end
    end
  end
  
  m = length(b);
  if maxiter > m
    maxiter = m;
  end
  if method == "left"
    b0 = precond(b);  % Left preconditioning
  elseif method == "right"
    b0 = b;           % Right preconditioning
  elseif method == "flexible"
    b0 = b;           % FGMRES
  end
  beta = norm(b0);
  residual = beta;
  fprintf("beta = %f\n", beta);
  tol = tol*beta;
  if verbose
    if method == "left"
      fprintf("Static GMRES, reaching preconditioned tolerance %8.5f in %d maximum iterations.\n", tol, maxiter);
    elseif method == "right"
      fprintf("Static GMRES, reaching absolute tolerance %8.5f in %d maximum iterations.\n", tol, maxiter);
    elseif method == "flexible"
      fprintf("Flexible GMRES, reaching absolute tolerance %8.5f in %d maximum iterations.\n", tol, maxiter);
    end
  end
  Q = b0/beta;
  omegaN = 1;
  H = [];
  if method == "flexible"
    Z = [];
  end
  residuals = [];

  for n = 1:maxiter

    % Arnoldi iteration
    if isa(A, 'function_handle')
      if method == "left"
	vec = precond(A(Q(:,n)));
      elseif method == "right"
	vec = A(precond(Q(:,n)));
      elseif method == "flexible"
	Z(:,n) = precond(Q(:,n)); % Forms basis for preconditioned Krylov subspace (AM^{-1})
	vec = A(Z(:,n));
      end
    else
      if method == "left"
	vec = precond(A*Q(:,n));
      elseif method == "right"
	vec = A*precond(Q(:,n));
      elseif method == "flexible"
	Z(:,n) = precond(Q(:,n)); % Forms basis for preconditioned Krylov subspace (AM^{-1})
	vec = A*Z(:,n);
      end
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
    % Efficient computation of residual
    residual = abs(t)*residual;
    if verbose
      % Residual equivalent to computing the following:
      e1 = [1; zeros(n,1)];
      y = H\(beta*e1);
      residual2 = norm(beta*e1 - H*y);
      if method == "left"
	x = Q*y;
      elseif method == "right"
	x = precond(Q*y);
      elseif method == "flexible"
	x = Z*y;
      end
      if isa(A, 'function_handle')
	res = b - A(x);
      else
	res = b - A*x;
      end
      if method == "left"
	residual3 = norm(precond(res));
      elseif method == "right"
	residual3 = norm(res);
      elseif method == "flexible"
	residual3 = norm(res);
      end
      fprintf("it: %4d, hN = %8.6f, res = %7.5f\n", n, hN, residual/beta);
      fprintf("||res1|| = %f, ||res2|| = %f, ||res3|| = %f\n", residual, residual2, residual3);
    end
    residuals = [residuals residual];
    if residual < tol
      fprintf("it: %4d, converging because residual %f < tol %f\n", n, residual, tol);
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
  if method == "left"
    x = Q * ( H\(beta*e1) );
  elseif method == "right"
    x = Q * ( H\(beta*e1) );
    x = precond(x);
  elseif method == "flexible"
    x = Z * ( H\(beta*e1) );
  end
  
end
