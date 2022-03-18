% From a set of matrices saved by Will's MFEM code, solve Ax=b with FGMRES preconditioned with subiteration
function mfem_subiteration_driver(mfem_dt, dt, tol, nsubiter, subtol_factor, global_precond_type, h, p, Lx, variable)

  if nargin < 10; variable = false; end;
  if nargin < 9; Lx = 1; end;
  if nargin < 8; p = 3; end;
  if nargin < 7; h = 1e-3; end;
  if nargin < 6
    global_precond_type = "jacobi";
  else
    if ~ismember(global_precond_type, ["jacobi", "mass_inv"])
      fprintf("ERROR: global preconditioner type not one of the acceptable types!\n");
      return;
    end
  end
  if nargin < 5; subtol_factor = 1.0; end;
  if nargin < 4; nsubiter = 500; end;
  if nargin < 3; tol = 1e-8; end;
  if nargin < 2; dt = 1e-3; end;
  if nargin < 1; mfem_dt = 1e-3; end

  if Lx == -1
    msh_name = sprintf("aniso_p%d_h%.0e", p, h);
  else
    msh_name = sprintf("aniso_p%d_h%.0e_Lx%d", p, h, Lx);
  end
  wr_dir = "/scratch/mfranco/2021/wr-les-solvers/";
  msh_dir = wr_dir+"meshes/";
  bl_file = msh_dir+msh_name+"bl.mat";
  msh_file = msh_dir+msh_name+".h5";
  if variable
    results_dir = wr_dir+"results/mfem_"+msh_name+"/variable/";
  else
    results_dir = wr_dir+"results/mfem_"+msh_name+"/";
  end
  mass_dir = results_dir+"mass/";

  fprintf("Running mfem_subiteration_driver(dt=%.1e) on mesh %s, nsubiter=%d.\n", dt, msh_name, nsubiter);

  A = load(results_dir+"A.mat", "-ascii");
  M = load(mass_dir+"Mass.mat", "-ascii");
  b = load(mass_dir+"residual.mat", "-ascii");
  [J, Ms, JD, Dij, b] = mfem_extract_matrices(A, M, b, msh_file, mfem_dt);
  
  bl_elems = freadarray(bl_file);
  bl_elems = uint64(bl_elems(:));

  %test_mfem_matrices(J, Ms, JD, Dij, b, dt);
  %Atimes = @(x) mfem_time_dependent_jacobian(J, Ms, dt, x);
  %test_matvec(Atimes, b, mass_dir);
  %return;

  subiteration_test(J, Ms, JD, Dij, b, bl_elems, dt, tol, nsubiter, subtol_factor, global_precond_type, true);
  
end

function [J, Ms, JD, Dij, b] = mfem_extract_matrices(A, M, b, msh_file, mfem_dt)
  msh = h5freadstruct(msh_file);
  nt = size(msh.p1,3);
  nlocal = size(msh.p1,1);
  
  Asparse = sparse(A(:,1), A(:,2), A(:,3));
  nlocal2 = size(Asparse,1)/nt;
  assert(nlocal == nlocal2, "mesh (%d) does not align with matrices (%d)!", nlocal, nlocal2);
  assert(nlocal*nt == size(Asparse,1) && nlocal*nt == size(Asparse,2), "A matrix does not fully span space");
  Msparse = sparse(M(:,1), M(:,2), M(:,3));
  assert(nlocal*nt == size(Msparse,1) && nlocal*nt == size(Msparse,2), "M matrix does not fully span space");
  J = Msparse./mfem_dt - Asparse; % Also sparse
  %b = b.*mfem_dt;
  % Extract block-diagonal information
  Ms = zeros(nlocal, nlocal, nt);
  JD = zeros(nlocal, nlocal, nt);
  Dij = zeros(nlocal*nlocal*nt,2);
  nnz = 0;
  for it = 1:nt
    offset1 = (it-1)*nlocal+1;
    offset2 = it*nlocal;
    Ms(:,:,it) = full(Msparse(offset1:offset2, offset1:offset2));
    JD(:,:,it) = full(J(      offset1:offset2, offset1:offset2));
    temp_indices = repmat(offset1:offset2.', 1, nlocal);
    Dij(nnz+1:nnz+nlocal*nlocal,1) = temp_indices(:);
    temp_indices = repmat(offset1:offset2, nlocal, 1);
    Dij(nnz+1:nnz+nlocal*nlocal,2) = temp_indices(:);
    nnz = nnz + nlocal*nlocal;
  end
  
end

% Tests that the MFEM matrices extracted pass a sanity check
function test_mfem_matrices(J, Ms, JD, Dij, b, dt)
  Atimes = @(x)mfem_time_dependent_jacobian(J, Ms, dt, x);
  diagA = Ms - dt*JD;
  disp(size(diagA))
  disp(Dij(1:20,:))
  disp(diagA(Dij(1:20,1)+(Dij(1:20,2)-ones(20,1))*size(diagA,1)))
  e=@(k,n) [zeros(k-1,1);1;zeros(n-k,1)];
  e_1 = e(1,size(J,1));
  first_row = Atimes(e_1);
  e_2 = e(2,size(J,1));
  second_row = Atimes(e_2);
  disp([first_row(1:20) second_row(1:20)]);
end

% Test that matvec done in Matlab is same as in MFEM C++
function test_matvec(Atimes, b, mass_dir)
  Ab1 = Atimes(b);
  Ab2 = load(mass_dir+"benchmark_matvec.mat", "-ascii");

  norm(b)
  norm(Ab1)
  norm(Ab2)
  fprintf("diff = %f\n", norm(Ab1-Ab2));
end

