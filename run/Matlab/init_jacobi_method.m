% Initialize block-diagonal Jacobi method for use as a solver
function solver = init_jacobi_method(A, diagA, b)
  % First LU decompose block diagonal for inverses used in Jacobi method
  nt = size(diagA,3);
  Dinvs = cell(nt,1);
  for it = 1:nt
    Dinvs{it} = decomposition(diagA(:,:,it), 'lu');
  end
  solver = @(x) evaluate_jacobi(Dinvs, A, b, x);
end

% Evaluate block diagonal Jacobi method applied to vector x
% y = x + D\(b-A*x) == D\(b-(L+U)x)
function y = evaluate_jacobi(Dinvs, A, b, x)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  if isa(A, 'function_handle')
    y1 = b - A(x);
  else
    y1 = b - A*x;
  end
  y2 = reshape(y1, nlocal, nt);
  for it = 1:nt
    y2(:,it) = Dinvs{it}\y2(:,it);
  end
  %if weighted
  %  y2 = (2.0/3.0)*y2; % Weighted Jacobi method
  %end
  y = reshape(y2, nlocal*nt, 1);
  y = y+x;
end
