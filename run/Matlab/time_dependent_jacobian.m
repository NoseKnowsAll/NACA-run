% Evaluate (Ms \otimes I - dt*J) * x
function y = time_dependent_jacobian(J, Ms, dt, x)
  y1 = -dt*J*x;
  nlocal = size(Ms,1);
  nt = size(Ms,3);
  y2 = reshape(y1, nlocal, nt);
  x2 = reshape(x,  nlocal, nt);
  for it = 1:nt
    y2(:,it) = y2(:,it) + Ms(:,:,it)*x2(:,it);
  end
  y = reshape(y2, nlocal*nt,1);
end