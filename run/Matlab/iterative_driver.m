% Test the iterative_method solver
function iterative_driver(n)
  if nargin < 1
    n = 100;
  end
  b = randn(n,1);
  tol = 1e-5;
  maxiter = 80;
  verbose = true;

  for itest = 1:3
    [A,diagA] = test_mat(itest, n);
    iteration = init_jacobi_method(A, diagA, b);
    x0 = randn(n,1);

    [x,iter,residuals] = iterative_method(A, b, x0, tol, maxiter, iteration, verbose);
  end
end

% Given a specific test ID, returns the corresponding test matrix 
function [A, diagA] = test_mat(test, n)
  if test == 1
    Ai = 1:n;
    Aj = 1:n;
    Ax = 1:n;
    A = sparse(Ai, Aj, Ax);
    diagA = zeros(1,1,n);
    diagA(1,1,:) = 1:n;
  elseif test == 2
    nlocal = 2;
    diagA = randn(nlocal,nlocal,n/nlocal);
    for i = 1:n/nlocal
      diagA(:,:,i) = diagA(:,:,i)'*diagA(:,:,i); % Ensure it's invertible by making it SPD
    end
    A = @(x)evaluate_diagonal_matrix(diagA, x);
  elseif test == 3
    nlocal = 5;
    diagA = randn(nlocal,nlocal,n/nlocal);
    for i = 1:n/nlocal
      diagA(:,:,i) = diagA(:,:,i)'*diagA(:,:,i); % Ensure it's invertible by making it SPD
    end
    % Off-diagonal will have nlocal nonzeros per row, located randomly
    nnz = nlocal*n;
    offAi = (1:n).*ones(nlocal,1);
    offAi = offAi(:);
    offAj = randi(n-nlocal, nlocal, n);
    for i = 1:n
      begin = floor((i-1)/nlocal)*nlocal;
      valid_indices = [1:begin begin+nlocal+1:n];
      for j = 1:nlocal
	offAj(j,i) = valid_indices(offAj(j,i));
      end
    end
    offAj = offAj(:);
    offAv = randn(nnz, 1)./(3*nnz); % Might get extremely unlikely rand thats 3std above mean 0 => singular matrix
    offA = sparse(offAi, offAj, offAv, n, n);
    A = @(x)mult_matrix(diagA, offA, x);
  else
    error("Unknown test!\n");
  end
end

% A = diagA+offA, where offA is a sparse matrix and diagA is block-diagonal of size (nlocal, nlocal, nt)
function y = mult_matrix(diagA, offA, x)
  nt = size(diagA, 3);
  nlocal = size(diagA, 1);
  y1 = offA*x;
  y2 = reshape(y1, nlocal, nt);
  x2 = reshape(x, nlocal, nt);
  for it = 1:nt
    y2(:,it) = y2(:,it) + diagA(:,:,it)*x2(:,it);
  end
  y = reshape(y2, nlocal*nt, 1);
end
