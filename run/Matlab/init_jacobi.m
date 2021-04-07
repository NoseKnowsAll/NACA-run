% Initialize Jacobi method for use as a preconditioner
function precond = init_jacobi(JD, A, b)
  % First LU decompose block diagonal for inverses used in Jacobi method
  nt = size(JD,3);
  Dinvs = cell(nt,1);
  for it = 1:nt
    Dinvs{it} = decomposition(JD(:,:,it), 'lu');
  end
  precond = @(x) evaluate_jacobi(Dinvs, A, b, x);
end

% Evaluate block diagonal Jacobi method applied to vector x
% y = x + D\(b-A*x) == D\(b-(L+U)x)
function y = evaluate_jacobi(Dinvs, A, b, x)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  y1 = b - A(x);
  y2 = reshape(y1, nlocal, nt);
  for it = 1:nt
    y2(:,it) = Dinvs{it}\y2(:,it);
  end
  % y2 = (2.0/3.0)*y2; % Weighted Jacobi method, only for MGPC
  y = reshape(y2, nlocal*nt, 1);
  y = y+x;
end
