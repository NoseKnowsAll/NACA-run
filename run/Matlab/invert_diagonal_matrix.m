% Evaluate (JD \otimes I) \ b
function x = invert_diagonal_matrix(JD, b)
  nlocal = size(JD,1);
  nt = size(JD,3);

  x2 = zeros(nlocal, nt);
  b2 = reshape(b, nlocal,nt);
  for it = 1:nt
    x2(:,it) = JD(:,:,it)\b2(:,it);
  end
  x = reshape(x2, nlocal*nt,1);
end
