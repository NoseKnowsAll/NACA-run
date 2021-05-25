% Driver script for testing various GMRES applications/options
function gmres_driver()

  msh_name = "naca_v2_p3_r12";
  bl_file = "/scratch/mfranco/2021/naca/meshes/"+msh_name+"bl.mat";
  msh_file = "/scratch/mfranco/2021/naca/run/partitioned/"+msh_name+".h5";
  results_dir = "/scratch/mfranco/2021/naca/run/results/"+msh_name+"/";
  mass_dir = results_dir+"mass/";
  dt  = 1e-3;

  %msh = h5freadstruct(msh_file);
  J   = freadjac(results_dir);
  JD  = freadarray(results_dir+"Dv.mat");
  Ms  = freadarray(mass_dir+"Dv.mat");
  b   = freadarray(results_dir+"mass/residual.mat");
  b   = b(:);
  areas = freadarray(mass_dir+"areas.mat");
  areas = areas(:);

  %Atimes = @(x)evaluate_eigenvalue_matrix(J, Ms, x);
  Atimes  = @(x)time_dependent_jacobian(J, Ms, dt, x);
  DA = Ms-dt*JD;
  
  precond = init_jacobi(DA, Atimes, b);
  %precond = init_mass_inv(Ms);
  %precond = @(x) x;
  
  maxiter = 500;
  restart = 500;
  tol = 1e-4;
  fprintf("GMRES initialized successfully\n");
  t_start = tic;
  %maxiter = 1; [x, flag, rel_res, iter, residuals] = gmres(Atimes, b, restart, tol, maxiter, precond);
  %maxiter = 1; [x, flag, rel_res, iter, residuals] = gmres(Atimes, b, restart, tol, maxiter);
  %disp(flag);

  %[x, iter, residuals] = cpp_gmres(Atimes, b, tol, maxiter, restart, precond, true);
  
  [x, iter, residuals] = static_gmres(Atimes, b, tol, maxiter, precond, "flexible", true);
  %[x, iter, residuals] = wiki_gmres(Atimes, b, tol, maxiter, precond, true);
  %scalings = 1/areas; [x, iter, residuals] = adaptive_gmres(Atimes, b, scalings, tol, maxiter, precond, "flexible", true);
  %fwritearray("adaptive_res.mat", residuals);

  t_gmres = toc(t_start);
  
  fprintf("total iterations: %d\n", iter);
  fprintf("time: %6.2f\n", t_gmres);
  fprintf("||Ax-b|| = %f\n", norm(Atimes(x)-b));
  
end

% Ensure that time_dependent_jacobian evaluates correctly
function test_time_dependent(results_dir, mass_dir, dt)
  J   = freadjac(results_dir);
  JD  = freadarray(results_dir+"Dv.mat");
  JO  = freadarray(results_dir+"Ov.mat");
  Dij = freadarray(results_dir+"Dij.mat");
  Oij = freadarray(results_dir+"Oij.mat");
  Ms  = freadarray(mass_dir+"Dv.mat");

  Atimes  = @(x)time_dependent_jacobian(J, Ms, dt, x);
  DA = Ms-dt*JD;

  ni = size(JD,1)*size(JD,3);
  Ai = double([Dij(1,:).'; Oij(1,:).'])+1; % MATLAB is 1-indexed
  Aj = double([Dij(2,:).'; Oij(2,:).'])+1; % MATLAB is 1-indexed
  Av = [DA(:); -dt*JO(:)];
  A = sparse(Ai, Aj, Av, ni, ni);

  test_x = randn(ni,1);
  y1 = Atimes(test_x);
  y2 = A*test_x;
  norm(y1)
  norm(y2)
  fprintf("diff = %f\n", norm(y1-y2));
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

% Test that jacobi iteration actually computes inverse of diagonal
function test_jacobi(results_dir, mass_dir, dt, b)

  J   = freadjac(results_dir);
  JD  = freadarray(results_dir+"Dv.mat");
  Ms  = freadarray(mass_dir+"Dv.mat");
  
  %Atimes = @(x) evaluate_diagonal_matrix(JD, x);
  Atimes  = @(x) time_dependent_jacobian(J, Ms, dt, x);

  DA = Ms-dt*JD;
  precond = init_jacobi(DA, Atimes, b);
  
  x3 = freadarray(mass_dir+"benchmark_jacobi.mat");
  x3 = x3(:);
  x4 = freadarray(mass_dir+"benchmark_unweighted_jacobi.mat");
  x4 = x4(:);

  %x = zeros(size(b));
  x = precond(b);

  x2 = invert_diagonal_matrix(JD, b);

  norm(x)
  norm(x2)
  norm(x3)
  norm(x4)
  fprintf("diff12 = %f\n", norm(x2-x));
  fprintf("diff13 = %f\n", norm(x3-x));
  fprintf("diff14 = %f\n", norm(x4-x));
  fprintf("diff23 = %f\n", norm(x3-x2));
  fprintf("diff24 = %f\n", norm(x4-x2));
  fprintf("diff34 = %f\n", norm(x4-x3));
  
end

% Tests whether Atimes(x) == JD*x
function test_diagonal_theory(results_dir, mass_dir, dt, b)
  
  J   = freadjac(results_dir);
  JD  = freadarray(results_dir+"Dv.mat");
  Ms  = freadarray(mass_dir+"Dv.mat");
  
  %Atimes = @(x) evaluate_diagonal_matrix(JD, x);
  Atimes  = @(x) time_dependent_jacobian(J, Ms, dt, x);

  DA = Ms-dt*JD;
  precond = init_jacobi(DA, Atimes, b);

  x = b;
  x2 = precond(Atimes(x));
  x = b;
  x3 = invert_diagonal_matrix(JD, Atimes(x));
  
  norm(x)
  norm(x2)
  norm(x3)
  fprintf("diff12 = %f\n", norm(x2-x));
  fprintf("diff13 = %f\n", norm(x3-x));
  fprintf("diff23 = %f\n", norm(x3-x2));
end

% Tests that backslash actually computes A\b
function test_backslash(results_dir, mass_dir, dt, b)
  J   = freadjac(results_dir);
  JD  = freadarray(results_dir+"Dv.mat");
  JO  = freadarray(results_dir+"Ov.mat");
  Dij = freadarray(results_dir+"Dij.mat");
  Oij = freadarray(results_dir+"Oij.mat");
  Ms  = freadarray(mass_dir+"Dv.mat");

  Atimes  = @(x)time_dependent_jacobian(J, Ms, dt, x);
  DA = Ms-dt*JD;

  ni = size(JD,1)*size(JD,3);
  Ai = double([Dij(1,:).'; Oij(1,:).'])+1; % MATLAB is 1-indexed
  Aj = double([Dij(2,:).'; Oij(2,:).'])+1; % MATLAB is 1-indexed
  Av = [DA(:); -dt*JO(:)];
  A = sparse(Ai, Aj, Av, ni, ni);

  Atimes  = @(x)time_dependent_jacobian(J, Ms, dt, x);
  DA = Ms-dt*JD;
  
  precond = init_jacobi(DA, Atimes, b);
  %precond = @(x) x;

  %x = A\b;
  %norm(x)
  norm(b)
  %norm(A*x-b)
  x2 = gmres(A, b, [], 1e-4, 10, precond);
  norm(x2)
  norm(A*x2-b)
  
end
