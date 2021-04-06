% First run eigenvalue.cpp to store the appropriate (Jacobian) matrices in a specified directory results_dir
% Then call this script to compute information on eigenvalues of matrix (ignoring boundary layer)

msh_name = "naca_v2_p3_r12";
bl_file = "/scratch/mfranco/2021/naca/meshes/"+msh_name+"bl.mat";
msh_file = "/scratch/mfranco/2021/naca/run/partitioned/"+msh_name+".h5";
results_dir = "/scratch/mfranco/2021/naca/run/results/"+msh_name+"/";
mass_dir = results_dir+"mass/";

msh = h5freadstruct(msh_file);
J_nonbl = init_jac_nonbl(msh, results_dir, bl_file);
Ms = freadarray(mass_dir+"Dv.mat");
%M_total = freadjac(mass_dir);
%figure(2);
%spy(M_total);
%return;
explore_eigenvalues(J_nonbl, Ms, msh_name);

% Evaluate ( M\otimes I ) \ J * x
function y = eval_matrix(J, M, x)
  y1 = J * x;
  nlocal = size(M,1);
  nt = size(M,3);
  y2 = reshape(y1, nlocal, nt);
  for it = 1:nt
    y2(:,it) = M(:,:,it)\y2(:,it);
  end
  y = reshape(y2, nlocal*nt,1);
end

% Print the n_eigs top magnitude eigenvalues of eval_matrix and spy Jacobian matrix
function explore_eigenvalues(J, Ms, msh_name, n_eigs)
  if nargin < 4
    n_eigs = 100;
  elseif nargin < 3
    msh_name="msh";
  end
  
  lambda = eigs(@(x)eval_matrix(J, Ms, x), size(J,1), n_eigs);
  fprintf("Max abs lambda: \n");
  disp(lambda(1:6));

  figure(1);
  spy(J);

  figure(2);
  plot(real(lambda), imag(lambda), "r*");
  savefig(msh_name+"_J_eigs.fig");
end

% Initialize J, but zero out the matrix for all rows corresponding to the boundary layer elements
function J_nonbl = init_jac_nonbl(msh, dir, bl_file)
  
  has_offdiag = isfile(dir+"Oij.mat");
  D = freadarray(dir+"Dv.mat");
  Dij = freadarray(dir+"Dij.mat");
  if has_offdiag
    O = freadarray(dir+"Ov.mat");
    Oij = freadarray(dir+"Oij.mat");
  else
    O = zeros(1,0);
    Oij = zeros(2,0);
  end
  
  ns = size(msh.p1,1);
  nt = size(msh.p1,3);
  nc = size(D,1)/ns;
  ni = size(D,1)*size(D,3);

  % Zero out everything in boundary layer
  bl_elems = freadarray(bl_file);
  %bl_elems = setdiff(1:nt, bl_elems); % TODO: Doing opposite of what we want right now
  non_bls  = setdiff(1:nt, bl_elems); 
  D(:,:,bl_elems) = 0;
  O(:,:,:,bl_elems) = 0;
  % Have to also zero out connections with neighbors of boundary layer
  for it = non_bls
    for j = 1:size(msh.t2t,1)
      it2 = msh.t2t(j,it);
      if ismember(it2,bl_elems)
	O(:,:,j,it) = 0;
      end
    end
  end

  Mi = double([Dij(1,:).'; Oij(1,:).'])+1; % MATLAB is 1-indexed
  Mj = double([Dij(2,:).'; Oij(2,:).'])+1; % MATLAB is 1-indexed
  M = [D(:); O(:)];

  J_nonbl = sparse(Mi, Mj, M, ni, ni);
end
