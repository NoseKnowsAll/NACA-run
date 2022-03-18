% Prepares plots of region-wise errors
function errors = variable_diffusion_driver(maxiter, dir)
  if nargin < 1; maxiter = 20; end;
  if nargin < 2; dir = "/scratch/mfranco/2021/naca/run/results/Matlab/"; end;
  if nargin < 3; msh_file = "/scratch/mfranco/2021/wr-les-solvers/meshes/aniso_p3_h1e-01.h5"; end;

  msh = h5freadstruct(msh_file);
  attr = cell(5,1); % Hardcode the attributes for this mesh
  attr{1} = 1:106;
  attr{2} = 107:120;
  attr{3} = 121:134;
  attr{4} = 135:148;
  attr{5} = 149:232;
  inner   = attr{3};
  overlap = [attr{[2,4]}];
  outer   = [attr{[1,5]}];

  errors = zeros(maxiter,3);

  for it = 1:maxiter
    iter = num2str(it, "%03.f");
    filename = sprintf("%serror_variable_it%s.mat", dir, iter);
    err = freadarray(filename);
    err_in   = norm(err(inner), Inf);
    err_over = norm(err(overlap), Inf);
    err_out  = norm(err(outer), Inf);
    errors(it,:) = [err_in, err_over, err_out];
  end

  semilogy(1:maxiter, errors(:,1), 'k*-', 1:maxiter, errors(:,2), 'r*-', 1:maxiter, errors(:,3), 'b*-');
end
