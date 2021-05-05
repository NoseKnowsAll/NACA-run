% "Official" GMRES version adapted from wikipedia
function [x, iter, residuals] = wiki_gmres(A, b, tol, maxiter, precond, verbose)

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

  %r = precond(b); % Left preconditioning
  r = b; % Right preconditioning
  b_norm = norm(b);
  residual = norm(r)/b_norm;
  sn = zeros(maxiter, 1);
  cs = zeros(maxiter, 1);
  residuals = [residual];
  Q(:,1) = r/norm(r);
  beta = zeros(maxiter, 1);
  beta(1) = norm(r)*1;

  if verbose
    fprintf("Wikipedia GMRES, reaching relative tolerance %.1e in %d maximum iterations.\n", tol, maxiter);
  end

  for n = 1:maxiter

    [H(1:n+1, n) Q(:,n+1)] = arnoldi(A, Q, precond, n);
    hN = H(n+1,n);
    [H(1:n+1, n) cs(n) sn(n)] = apply_givens_rotation(H(1:n+1,n), cs, sn, n);

    % Update residual vector
    beta(n+1) = -sn(n)*beta(n);
    beta(n)   = cs(n)*beta(n);
    residual  = abs(beta(n+1))/b_norm;
    residuals = [residuals residual];

    if verbose
      fprintf("it: %4d, hN = %8.6f, res = %7.5f\n", n, hN, residual);
    end
    if residual < tol
      break;
    end
    
  end

  % Now that convergence has been met, retrieve solution x
  iter = n;
  x = Q(:,1:iter) * ( H(1:iter,1:iter)\beta(1:iter) );
  x = precond(x); % Right preconditioning

end

% Perform an Arnoldi iteration
function [h, q] = arnoldi(A, Q, precond, n)
  if isa(A, 'function_handle')
    %q = precond(A(Q(:,n))); % Left preconditioning
    q = A(precond(Q(:,n))); % Right preconditioning
  else
    %q = precond(A*Q(:,n)); % Left preconditioning
    q = A*precond(Q(:,n)); % Right preconditioning
  end
  h = zeros(n+1,1);
  for i = 1:n
    h(i) = q' * Q(:,i);
    q = q - h(i)*Q(:,i);
  end
  h(n+1) = norm(q);
  q = q / h(n+1);
end

% Create a Givens rotation matrix based on 2 values so that G*[v1; v2] will zero out v2
function [cs, sn] = givens_rotation(v1, v2)
  denom = sqrt(v1^2 + v2^2);
  cs = v1/denom;
  sn = v2/denom;
end

% Apply specified Givens rotation matrix to specified row of H matrix
function [h, cs_k, sn_k] = apply_givens_rotation(h, cs, sn, k)
  % apply to ith column of H
  for i = 1:k-1
    temp   =  cs(i) * h(i) + sn(i) * h(i+1);
    h(i+1) = -sn(i) * h(i) + cs(i) * h(i+1);
    h(i)   = temp;
  end

  % update the next sin cos values for rotation
  [cs_k sn_k] = givens_rotation(h(k), h(k+1));

  % eliminate H(i + 1, i)
  h(k) = cs_k * h(k) + sn_k * h(k+1);
  h(k+1) = 0.0;
end
