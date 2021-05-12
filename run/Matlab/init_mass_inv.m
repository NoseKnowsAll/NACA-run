% Initialize preconditioner to be M^{-1}x
function precond = init_mass_inv(Ms)
  nt = size(Ms,3);
  Minvs = cell(nt,1);
  for it = 1:nt
    Minvs{it} = decomposition(Ms(:,:,it), 'lu');
  end
  precond = @(x) evaluate_mass_inv(Minvs, x);
end

% Evaluate block diagonal mass-inverse applied to x
function y = evaluate_mass_inv(Minvs, x)
  nt = size(Minvs,1);
  nlocal = Minvs{1}.MatrixSize(2);
  x2 = reshape(x, nlocal, nt);
  y2 = x2;
  for it = 1:nt
    y2(:,it) = Minvs{it}\y2(:,it);
  end
  y = reshape(y2, nlocal*nt, 1);
end
