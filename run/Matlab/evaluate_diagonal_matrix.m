% Evaluate (JD \otimes I) * x
function y = evaluate_diagonal_matrix(JD, x)
  nlocal = size(JD,1);
  nt = size(JD,3);

  y2 = zeros(nlocal, nt);
  x2 = reshape(x, nlocal,nt);
  for it = 1:nt
    y2(:,it) = JD(:,:,it)*x2(:,it);
  end
  y = reshape(y2, nlocal*nt,1);
end
