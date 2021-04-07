% Driver script for testing various GMRES applications/options
function gmres_driver()

  msh_name = "naca_v2_p3_r12";
  bl_file = "/scratch/mfranco/2021/naca/meshes/"+msh_name+"bl.mat";
  msh_file = "/scratch/mfranco/2021/naca/run/partitioned/"+msh_name+".h5";
  results_dir = "/scratch/mfranco/2021/naca/run/results/"+msh_name+"/";
  mass_dir = results_dir+"mass/";

  msh = h5freadstruct(msh_file);
  J   = freadjac(results_dir);
  JD  = freadarray(results_dir+"Dv.mat");
  Ms  = freadarray(mass_dir+"Dv.mat");
  b   = freadarray(results_dir+"residual.mat");
  b   = b(:);
  dt  = 1e-3;

  %Atimes = @(x)evaluate_matrix(J, Ms, x);
  Atimes  = @(x)time_dependant_jacobian(J, Ms, dt, x);
  precond = init_jacobi(JD, Atimes, b);
  %precond = @(x) x;
  maxiter = 2000;
  restart = 200;
  tol = 1e-4;
  fprintf("GMRES initialized successfully\n");
  t_start = tic;
  [x, iter, residuals] = static_gmres(Atimes, b, tol, maxiter, precond, true);
  t_gmres = toc(t_start);
  
  fprintf("total iterations: %d\n", iter);
  fprintf("time: %6.2f\n", t_gmres);
  disp(residuals);

end

% Evaluate ( Ms\otimes I ) \ J * x
function y = evaluate_matrix(J, Ms, x)
  y1 = J * x;
  nlocal = size(Ms,1);
  nt = size(Ms,3);
  y2 = reshape(y1, nlocal, nt);
  for it = 1:nt
    y2(:,it) = Ms(:,:,it)\y2(:,it);
  end
  y = reshape(y2, nlocal*nt,1);
end

% Evaluate (Ms \otimes I - dt*J) * x
function y = time_dependant_jacobian(J, Ms, dt, x)
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
