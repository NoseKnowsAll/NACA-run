% First run naca_subiteration_driver(1e-3, 1e-8, 500, "jacobi") with residuals output to ../Matlab/

function plot_residual_iterations(maxiter, components)

  if nargin < 2
    components = 1;
    if nargin < 1
      maxiter = 12;
    end
  end
  
  msh_name = "naca_v2_p3_r12";
  bl_file = "/scratch/mfranco/2021/naca/meshes/"+msh_name+"bl.mat";
  msh_file = "/scratch/mfranco/2021/naca/run/partitioned/"+msh_name+".h5";
  results_dir = "/scratch/mfranco/2021/naca/run/results/Matlab/";
  residual_file = results_dir+"residual_it";

  msh = h5freadstruct(msh_file);
  nc = 4; % 2D Navier-Stokes
  nlocal = size(msh.p1, 1);
  nt = size(msh.p1, 3);

  for it = 1:maxiter
    fprintf("Working on iteration %d...\n", it);
    % Read in residuals
    iterstr = num2str(it, "%03.f");
    res_file = sprintf("%s%s.mat", residual_file, iterstr);
    res = freadarray(res_file);
    % Consider only |res|
    res = reshape(abs(res), [nlocal, nc, nt, 1]);

    % Plot residuals and save to a fig file
    for ic = components
      figure((it-1)*nc + ic);
      dgplot(msh, res, ic, [], ceil(log2(3)), true);
      colorbar;
      fig_file = sprintf("%s%s-%d.fig", residual_file, iterstr, ic);
      savefig(fig_file);
    end
  end
  
end
