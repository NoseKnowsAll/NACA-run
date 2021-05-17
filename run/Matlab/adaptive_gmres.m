% GMRES on element-based matrix/vectors and adaptively ignore elements where answer has converged.
% nlocal is size of all DoFs related to one element.
% A can be a matrix s.t. A*x = b is the linear system to solve or
% A can be a function handle s.t. A(x) = A*x = b is linear system to solve.
% precond is a function handle s.t. precond(x) = P\x for use in solving
% (a) left-preconditioned system P\Au = P\b
% or (b) right-preconditioned system AP\u = b, u=Px
% Can specify preconditioning of GMRES method to be "left", "right", or "flexible".
% Default method: "flexible" => FGMRES.
function [x, iter, residuals] = adaptive_gmres(A, b, scalings, tol, maxiter, precond, method, verbose)
  
  % Initialization
  if nargin < 8
    verbose = false;
    if nargin < 7
      method = "flexible";
      if nargin < 6
	precond = @(x) x;
      end
    else
      if ~(ismember(method, ["left", "right", "flexible"]))
	fprintf("Invalid method specified for GMRES. Resetting to FGMRES...\n");
	method = "flexible";
      end
    end
  end
  if isa(scalings, 'number')
    nt = scalings;
  else
    nt = length(scalings);
  end
  if verbose
    fprintf("Adaptive GMRES, reaching local relative tolerances %.1e in %d maximum iterations.\n", tol, maxiter);
  end
  
  m = length(b);
  nlocal = m/nt;
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
  local_converge_iters = zeros(nt,1);
  local_residuals = local_norms(b0, nlocal);
  % TODO: What is the actual correct level of accuracy?
  tols = tol*local_residuals*scalings; % We need more accuracy for each element in order to reach global accuracy of static GMRES. 
  Q = b0/beta;
  omegaN = 1;
  H = [];
  Z = [];
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
    hN = norm(vec); % TODO: this value is still an unknown to me in true adaptive GMRES
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
    % Can no longer efficiently compute residual.
    % residual = abs(t)*residual;
    
    e1 = [1; zeros(n,1)];
    y = H\(beta*e1);
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
      res = precond(res);
    end
    fprintf("||res|| = %f\n", norm(res));
    local_residuals = local_norms(res, nlocal)*scalings;
    % TODO: Should be the following line actually, but the other way I don't have to worry about re-orderings
    % local_residuals = local_norms(res, nlocal, local_converge_iters == 0);
    %local_converge_iters(local_converge_iters == 0 && local_residuals < tols) = n;

    residuals = [residuals local_residuals];
    for i = 1:length(local_converge_iters)
      if local_converge_iters(i) == 0 && local_residuals(i) < tols(i)
        local_converge_iters(i) = n;
	fprintf("%d just converged.\n", i);
      end
    end
    n_unconverged = nnz(local_converge_iters == 0);
    if verbose
      fprintf("it: %4d, hN = %8.6f, unconverged = %d\n", n, hN, n_unconverged);
    end
    if n_unconverged == 0
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

% Computes a vector of 2-norms applied locally to each element's portion of vector.
% Optionally, only compute norms for a subset of elements determined by indices variable.
function lcl_norms = local_norms(vec, nlocal, indices)
  vec_shaped = reshape(vec, nlocal,[]);
  nt = size(vec_shaped,2);

  if nargin < 3
    indices = 1:nt;
  end

  lcl_norms = vecnorm(vec_shaped(:,indices)).';
end
