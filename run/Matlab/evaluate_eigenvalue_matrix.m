% Evaluate ( Ms\otimes I ) \ J * x
function y = evaluate_eigenvalue_matrix(J, Ms, x)
  y1 = J * x;
  nlocal = size(Ms,1);
  nt = size(Ms,3);
  y2 = reshape(y1, nlocal, nt);
  for it = 1:nt
    y2(:,it) = Ms(:,:,it)\y2(:,it);
  end
  y = reshape(y2, nlocal*nt,1);
end
