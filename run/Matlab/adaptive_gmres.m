% GMRES on element-based matrix, vectors and adaptively ignore elements where answer has converged
function [x, iter] = adaptive_gmres(A, b, tol, maxiter)
  
  % initialization
  m = length(b);
  if maxiter > m
    maxiter = m;
  end
  residual = norm(b);
  tol = tol*(1+residual);
  Q = b/residual;
  omegaN = 1;
  H = [];

  for n = 1:maxiter

    % Arnoldi iteration
    vec = A*Q(:,n);
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
    if (hN < tol)
      break;
    end
    residual = abs(t)*residual;
    if (residual < tol)
      break;
    end

    % Update for next iteration
    Q = [Q vec/hN];
    omegaN = [-t*omegaN; v];

  end

  % Now that convergence has been met, retrieve solution x
  iter = n;
  e1 = [1; zeros(iter,1)];
  x = Q * ( H\(norm(b)*e1) );
  
end
