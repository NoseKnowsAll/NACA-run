% Restarted FGMRES version adapted from MFEM C++ version
function [x, iter, residuals] = restarted_fgmres(A, b, x0, tol, restart, maxiter, precond, verbose)

  % Initialization
  if nargin < 7
    verbose = false;
    if nargin < 6
      precond = @(x) x;
    end
  end
  sz = length(b);
  m = restart;
  if maxiter > sz
    maxiter = sz;
  end
  if isempty(x0)
    x = zeros(size(b));
    r = b;
  else
    x = x0;
    if isa(A, 'function_handle')
      r = b - A(x);
    else
      r = b - A*x;
    end
  end
  beta = norm(r);
  final_norm = beta*tol;
  if beta <= final_norm
    residuals = beta;
    iter = 0;
    if verbose
      fprintf("Immediately converging because %8.2e < %8.2e\n", beta, final_norm);
    end
    return;
  end

  fprintf("Restarted FGMRES, reaching absolute tolerance %8.2e in %d,%d maximum iterations.\n", final_norm, restart, maxiter);
  
  H = zeros(m+1,m);
  cs = zeros(m+1,1);
  sn = zeros(m+1,1);
  v = zeros(sz,m+1);
  Z = zeros(sz,m+1);
  residuals = [];
  converged = false;

  j = 0;
  while j < maxiter
    v(:,1) = r/beta;
    s = zeros(m+1,1);
    s(1) = beta;
    for i = 1:m
      j = j+1;
      if j > maxiter
	break;
      end

      % FGMRES version of Arnoldi iteration
      Z(:,i) = precond(v(:,i));
      if isa(A, 'function_handle')
	r = A(Z(:,i));
      else
	r = A*Z(:,i);
      end
      for k = 1:i %TODO: Is this i+1?
	H(k,i) = r'*v(:,k);
	r = r - H(k,i)*v(:,k);
      end
      H(i+1,i) = norm(r);
      v(:,i+1) = r/H(i+1,i);

      % Apply Givens rotation to H matrix
      [H(1:i+1,i), s(i), s(i+1), cs(i), sn(i)] = apply_givens_rotation(H(1:i+1,i), s(1:i+1), cs, sn, i);
      residual  = abs(s(i+1)); %abs(beta(n+1))/b_norm;
      residuals = [residuals residual];
      if verbose
	fprintf("it: %4d/%4d, ||res|| = %8.2e\n", j, restart, residual);
      end

      % Check convergence
      if residual <= final_norm
	% TODO: Will's version immediately quits based on first i+1 values?
	x = x + Z(:,1:i) * (H(1:i+1,1:i)\s(1:i+1));
	converged = true;
	iter = j;
	if verbose
	  fprintf("it: %4d/%4d, converging because residual %8.2e < tol %8.2e\n", j, restart, residual, final_norm);
	end
	return;
      end
      
    end

    % Restart FGMRES for the next restarted FGMRES iteration
    x = x + Z(:,1:i-1) * (H(1:i,1:i-1)\s(1:i));
    if isa(A, 'function_handle')
      r = b - A(x);
    else
      r = b - A*x;
    end
    iter = j;
    beta = norm(r);
    
  end

  %fprintf("ERROR: Did not converge after %d iterations.\n", j);
  %x = x0;
  %iter = 0;
  %residuals = [];

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

% Given a rotation specified by cs and sn constants, apply it to (dx,dy)
function [dx, dy] = apply_rotation(dx, dy, cs, sn)
  temp =  cs * dx + sn * dy;
  dy   = -sn * dx + cs * dy;
  dx   = temp;
end

% Apply specified Givens rotation matrix to specified row of H matrix
function [h, sk, sk1, cs_k, sn_k] = apply_givens_rotation(h, s, cs, sn, k)
  % apply to ith column of H
  for i = 1:k-1
    [h(i), h(i+1)] = apply_rotation(h(i), h(i+1), cs(i), sn(i));
  end

  % update the next sin cos values for rotation
  [cs_k sn_k] = givens_rotation(h(k), h(k+1));

  % eliminate H(i + 1, i)
  [h(k), h(k+1)] = apply_rotation(h(k), h(k+1), cs_k, sn_k);

  % Update s for the next iteration
  [sk, sk1] = apply_rotation(s(k), s(k+1), cs_k, sn_k);
  
end
