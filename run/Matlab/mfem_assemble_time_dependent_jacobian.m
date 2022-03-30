% Fully assemble sparse matrix A = Ms/dt \otimes I - J
function A = mfem_assemble_time_dependent_jacobian(J, Ms, dt)
  nlocal = size(Ms, 1);
  nt = size(Ms, 3);
  [Ji, Jj, Jv] = find(J);
  Jv = -Jv;
  Di = zeros(nlocal*nlocal*nt,1);
  Dj = zeros(nlocal*nlocal*nt,1);
  Dv = Ms(:)/dt;
  nnz = 0;
  for it = 1:nt
    ii1 = nlocal*(it-1) + 1;
    iin = nlocal*(it);
    Di(nnz+1:nnz+nlocal*nlocal,1) = reshape(repmat(ii1:iin.', 1, nlocal), [],1);
    Dj(nnz+1:nnz+nlocal*nlocal,1) = reshape(repmat(ii1:iin,   nlocal, 1), [],1);
    nnz = nnz + nlocal*nlocal;
  end
  A = sparse([Ji; Di], [Jj; Dj], [Jv; Dv], size(J,1), size(J,2));
end
