% First run for instance naca_subiteration_driver(1e-3, 1e-8, 500, "jacobi") with residuals output to ../Matlab/
% maxiter = number of outer iterations that created residuals to view
% components = array specifying which components to display as separate figures
% msh_dir = string: directory containing mesh
% msh_name = string: mesh name without file extension
% res_name = string: residuals files all of form "$res_name_it%03.f.mat" in "../Matlab/"
function plot_residual_iterations(maxiter, components, msh_dir, msh_name, res_name)
  if nargin < 5
    res_name = "residual";
    if nargin < 4
      msh_name = "naca_v2_p3_r12";
      if nargin < 3
	msh_dir = "/scratch/mfranco/2021/naca/meshes/";
	if nargin < 2
	  components = 1;
	  if nargin < 1
	    maxiter = 12;
	  end
	end
      end
    end
  end
  close all;
  
  bl_file  = msh_dir+msh_name+"bl.mat";
  msh_file = msh_dir+msh_name+".h5";
  results_dir = "/scratch/mfranco/2021/naca/run/results/Matlab/";
  residual_file = results_dir+res_name+"_it";

  msh = h5freadstruct(msh_file);
  nnodes = size(msh.p1, 1);
  nt = size(msh.p1, 3);

  for it = 1:maxiter
    fprintf("Working on iteration %d...\n", it);
    % Read in residuals
    iterstr = num2str(it, "%03.f");
    res_file = sprintf("%s%s.mat", residual_file, iterstr);
    res = freadarray(res_file);
    nc = prod(size(res))/nt/nnodes;
    nlocal = nc*nnodes;
    res = reshape(res, [nlocal, nt]);
    % Consider only vecnorm(res)
    norms = vecnorm(reshape(res, [nlocal, nt]));
    norm_u = zeros(nnodes, nc, nt, 1);
    for i = 1:nt
      norm_u(:,:,i,1) = log10(norms(i));
    end
    %res = reshape(abs(res), [nnodes, nc, nt, 1]);

    % Plot residuals and save to a fig file
    for ic = components
      figure((it-1)*nc + ic);
      dgplot(msh, norm_u, ic, [], ceil(log2(3)), true);
      set(groot, 'DefaultTextInterpreter', 'none')
      title(sprintf("%s%s-%d", res_name+"_it", iterstr, ic));
      colorbar;
      fig_file = sprintf("%s%s-%d.fig", residual_file, iterstr, ic);
      savefig(fig_file);
    end
  end
  
end
