% Initialize Jacobi method for use as a preconditioner
% Note: Actually it's just block-diagonal D\rhs
function precond = init_jacobi(diagA)
  % First LU decompose block diagonal for inverses used in Jacobi method
  nt = size(diagA,3);
  Dinvs = cell(nt,1);
  for it = 1:nt
    Dinvs{it} = decomposition(diagA(:,:,it), 'lu');
  end
  precond = @(rhs) evaluate_jacobi(Dinvs, rhs);
end

% Evaluate block diagonal Jacobi method applied to vector x
% x = D\rhs
function x = evaluate_jacobi(Dinvs, rhs)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  
  x2 = zeros(nlocal, nt);
  rhs2 = reshape(rhs, nlocal, nt);
  for it = 1:nt
    x2(:,it) = Dinvs{it}\rhs2(:,it);
  end
  x = reshape(x2, nlocal*nt, 1);
end
