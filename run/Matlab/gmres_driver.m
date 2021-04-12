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
  b   = freadarray(results_dir+"mass/residual.mat");
  b   = b(:);

  dt  = 1e-3;

  %Atimes = @(x)evaluate_matrix(J, Ms, x);
  Atimes  = @(x)time_dependant_jacobian(J, Ms, dt, x);
  DA = Ms-dt*JD;
  
  precond = init_jacobi(DA, Atimes, b);
  %precond = @(x) x;
  
  maxiter = 500;
  restart = 500;
  tol = 1e-4;
  fprintf("GMRES initialized successfully\n");
  t_start = tic;
  %[x, flag, rel_res, iter, residuals] = gmres(Atimes, b, restart, tol, maxiter, precond);
  %disp(flag);
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

% Test that matvec done in Matlab is same as in C++
function test_matvec(Atimes, b, mass_dir)
  Ab1 = Atimes(b);
  Ab2 = freadarray(mass_dir+"benchmark_matvec.mat");
  Ab2 = Ab2(:);

  norm(b)
  norm(Ab1)
  norm(Ab2)
  fprintf("diff = %f\n", norm(Ab1-Ab2));
end

% Evaluate (JD \otimes I) * x
function y = evaluate_diagonal_matrix(JD, x)
  nlocal = size(JD,1);
  nt = size(JD,3);

  y2 = zeros(nlocal, nt);
  x2 = reshape(x, nlocal,nt);
  for it = 1:nt
    y2(:,it) = JD(:,:,it)*x2(:,it);
  end
  y = reshape(y2, nlocal*nt,1);
end

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

% Test that jacobi iteration actually computes inverse of diagonal
function test_jacobi(JD, b, mass_dir)
  Atimes = @(x) evaluate_diagonal_matrix(JD, x);
  precond = init_jacobi(JD, Atimes, b);
  x3 = freadarray(mass_dir+"benchmark_jacobi.mat");
  x3 = x3(:);

  x = zeros(size(b));
  x = precond(x);

  x2 = invert_diagonal_matrix(JD, b);

  norm(x)
  norm(x2)
  norm(x3)
  fprintf("diff12 = %f\n", norm(x2-x));
  fprintf("diff23 = %f\n", norm(x3-x2));
  fprintf("diff13 = %f\n", norm(x3-x));
end
