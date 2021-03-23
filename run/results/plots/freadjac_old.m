% Reads in a Jacobian matrix stored as 4 input files
% (Dij.mat, Dx.mat, Oij.mat, Ox.mat) created from
% the underlying data and dgjacindices(). Oij/Ox optional.
% dir == input directory containing these 4 files
% s_jac == output sparse matrix 
function s_jac = freadjac_old(dir)

if nargin < 1
  dir = "./"
end

has_offdiag = isfile(dir+"Oij.mat");

Dij = freadarray(dir+"Dij.mat");
Dx  = freadarray(dir+"Dx.mat");
if has_offdiag
  Oij = freadarray(dir+"Oij.mat");
  Ox  = freadarray(dir+"Ox.mat");
else
  Oij = zeros(2,0);
  Ox  = zeros(1,0);
end

ni = size(Dx,1)*size(Dx,3);

Mi = double([Dij(1,:); Oij(1,:)])+1; % MATLAB is 1-indexed
Mj = double([Dij(2,:); Oij(2,:)])+1; % MATLAB is 1-indexed
Mx = [Dx(:); Ox(:)];

s_jac = sparse(Mi, Mj, Mx, ni, ni);

end
