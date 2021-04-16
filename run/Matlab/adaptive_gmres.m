% GMRES on element-based matrix, vectors and adaptively ignore elements where answer has converged.
% nlocal is size of all DoFs related to one element.
% A can be a matrix s.t. A*x = b is the linear system to solve or
% A can be a function handle s.t. A(x) = A*x = b is linear system to solve.
% precond is a function handle s.t. precond(x) = P\x for use in solving
% left-preconditioned system P\Ax = P\b.
function [x, iter, residuals] = adaptive_gmres(A, b, nlocal, tol, maxiter, precond, verbose)
  
  % Initialization
  if nargin < 7
    verbose = false;
    if nargin < 6
      precond = @(x) x;
    end
  end
  if verbose
    fprintf("Adaptive GMRES, reaching local relative tolerances %.1e in %d maximum iterations.\n", tol, maxiter);
  end
  
  m = length(b);
  nt = m/nlocal;
  if maxiter > m
    maxiter = m;
  end
  b0 = precond(b);
  beta = norm(b0);
  local_converge_iters = zeros(nt,1);
  local_residuals = local_norms(b0, nlocal);
  tols = tol*local_residuals;
  Q = b0/beta;
  omegaN = 1;
  H = [];
  residuals = [];

  for n = 1:maxiter

    % Arnoldi iteration
    if isa(A, 'function_handle')
      vec = A(Q(:,n));
    else
      vec = A*Q(:,n);
    end
    vec = precond(vec);
    h = Q'*vec;
    vec = vec - Q*h;
    hN = norm(vec); % TODO: this value is still an unknown to me
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
    x = Q*y;
    if isa(A, 'function_handle')
      res = b - A(x);
    else
      res = b - A*x;
    end
    fprintf("||res|| = %f\n", norm(res));
    res = precond(res);
    local_residuals = local_norms(res, nlocal);
    % TODO: Should be this actually, but the other way I don't have to worry about re-orderings
    % local_residuals = local_norms(res, nlocal, local_converge_iters == 0);
    residuals = [residuals local_residuals];
    for i = 1:length(local_converge_iters)
      if local_converge_iters(i) == 0 && local_residuals(i) < tols(i)
        local_converge_iters(i) = n;
	fprintf("%d just converged.\n", i);
      end
    end
    %local_converge_iters(local_converge_iters == 0 && local_residuals < tols) = n;
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
  x = Q * ( H\(beta*e1) );
  
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
