function [x, iter, residuals] = cpp_gmres(A, b, x, tol, maxiter, restart, precond, verbose)

  % Initialization
  n = length(b);
  m = restart;
  if mod(maxiter,m) > 0
    maxiter = floor(maxiter/(m+1)) * m;
  end
  fprintf("Original ||b|| = %7.5f\n", norm(b));
  r = b;
  r = precond(r); % Left preconditioning
  nrm2b = norm(r);
  v = zeros(n,m+1); % == Q (orthogonal matrix spanning Krylov subspace)
  h = zeros(m+1,m); % Hessenberg matrix: Organization is different from Per's code
  y = zeros(m+1,1); % Solution of LLS problem. x = Q*y
  c = zeros(m,1);
  s = zeros(m,1);
  if isempty(x)
    x = zeros(n,1);
  end
  residuals = [];

  disp(maxiter);
  disp(tol);
  disp(nrm2b);

  disp(b(1:5));
  test = precond(b);
  disp(test(1:5));
  disp("Beginning outer iteration");

  % Outer iteration
  for j = 1:maxiter/m
    if isa(A, 'function_handle')
      r = A(x);
      disp(r(1:5));
      r = r - b;
      disp(r(1:5));
      r = precond(r);
      disp(r(1:5));
      %r = precond(A(x)-b); % Left preconditioning
      %r = A(precond(x)-b); % Right preconditioning
    else
      r = precond(A*x-b); % Left preconditioning
      %r = A*(precond(x)-b); % Right preconditioning
    end
    beta = norm(r);
    v(:,1) = r/beta;
    y(1) = beta;
    fprintf("\nv_0(1) = %7.5e\n", v(1,1));

    % Inner iteration
    for i = 1:m
      residuals = [residuals abs(y(i))/nrm2b];
      if abs(y(i)) < tol*nrm2b
	if verbose
	  fprintf("Inner iteration i=%d converging: %f < %f\n", i, abs(y(i)), tol*nrm2b);
	end
	break;
      end
      if verbose
	fprintf("%4d: %7.5f, ", i, abs(y(i)));
      end

      % Arnoldi iteration
      vi = v(:,i);
      v(:,i+1) = v(:,i);
      if isa(A, 'function_handle')
	v(:,i+1) = precond(A(v(:,i+1))); % Left preconditioning
	%v(:,i+1) = A(precond(v(:,i+1))); % Right preconditioning
      else
	v(:,i+1) = precond(A*v(:,i+1)); % Left preconditioning
	%v(:,i+1) = A*precond(v(:,i+1)); % Right preconditioning
      end
      fprintf("v_{i+1}(1) = %7.5e ", (v(1,i+1)));
      h(1:i,i) = v(:,1:i)'*v(:,i+1);
      v(:,i+1) = v(:,i+1) - v(:,1:i)*h(1:i,i);
      fprintf("v_{i+1}(1) = %7.5e ", (v(1,i+1)));
      hnew = norm(v(:,i+1));
      v(:,i+1) = v(:,i+1)/hnew;
      if verbose
	fprintf("hN = %7.5f\n", hnew);
      end

      % Givens rotation applied to ith column of H
      for k = 1:i-1
	tmp      = c(k)*h(k,i)-s(k)*h(k+1,i);
	h(k+1,i) = s(k)*h(k,i)+c(k)*h(k+1,i);
	h(k,i)   = tmp;
      end
      rd = h(i,i);
      dd = sqrt(rd*rd + hnew*hnew);
      c(i) = rd/dd;
      s(i) = -hnew/dd;
      h(i,i) = dd;

      y(i+1) = s(i)*y(i);
      y(i)   = c(i)*y(i);
      
    end

    % Now update the actual solution
    y(1:i-1) = h(1:i-1,1:i-1)\y(1:i-1);
    
    disp("y=");
    disp(y(1:5));
    x = x - v(:,1:i-1)*y(1:i-1);
    disp("x=");
    disp(x(1:5));
    residuals = [residuals abs(y(i))/nrm2b];
    if abs(y(i)) < tol*nrm2b || j == floor(maxiter/m) - 1
      break;
    end
    
  end

  iter = m*(j-1)+i-1;
  
end
