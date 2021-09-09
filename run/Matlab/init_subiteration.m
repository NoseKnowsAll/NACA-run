% Creates a subiteration preconditioner to solve A\b
function precond = init_subiteration(A, diagA, Ms, bl_elems, b, global_precond_type, tol, nsubiter)
  nt = size(diagA,3);
  nlocal = size(diagA,2);
  fprintf("Subiteration preconditioner with nsubiter=%d, nt=%d, size(bl_elems)=%d\n", nsubiter, nt, length(bl_elems));
  
  Dinvs = cell(nt,1);
  if global_precond_type == "jacobi"
    for it = 1:nt
      Dinvs{it} = decomposition(diagA(:,:,it), 'lu');
    end
  elseif global_precond_type == "mass_inv"
    for it = 1:nt
      Dinvs{it} = decomposition(Ms(:,:,it), 'lu');
    end
  end
  precond_global = @(rhs) apply_global_preconditioner(Dinvs, rhs);

  [A_bl, diagA_bl] = extract_suboperator(A, diagA, bl_elems);
  
  precond_bl = init_jacobi(diagA_bl);
  precond = @(rhs) evaluate_subiteration(A, A_bl, diagA_bl, precond_global, precond_bl, tol, nsubiter, bl_elems, nt, nlocal, rhs);
  
  global outer_iteration;
  outer_iteration = 0;
end

% Subiteration preconditioner is one application of x = P\rhs.
% First apply block diagonal inverse to entire region. x = D\rhs
% Then consider just the boundary layer elements specified by bl_elems.
% Compute boundary layer residual r_{bl} = rhs_{bl} - A_{bl}*x_{bl}
% Perform inner GMRES iteration to solve A_{bl}*e_{bl} = r_{bl} with initial guess x_{bl} to get e_{bl}
% Update x according to correction: x = x + e_{bl}
% Note: Only valid as a preconditioner within a broader FGMRES iteration
function x = evaluate_subiteration(A, A_bl, diagA_bl, precond_global, precond_bl, tol, nsubiter, bl_elems, nt, nlocal, rhs)

  x = precond_global(rhs);
  %test_mfem(x, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_step0.mat");
  global outer_iteration;
  outer_iteration = outer_iteration + 1;
  %if outer_iteration > 8 % TODO: ramped subiteration
  %  nsubiter = 0;
  %end
  %nsubiter = 0; % TODO: only global preconditioner
  if nsubiter <= 0
    return;
  end

  if outer_iteration == 1
    test_mfem(A(x), "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_Ax.mat");
  end
  
  % TODO: check if convergence changes when we compute true r_bl = rhs_bl - A_bl(x_bl)
  if isa(A, 'function_handle')
    r = rhs - A(x);
  else
    r = rhs - A*x;
  end
  if outer_iteration == 1
    test_mfem(-r, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_step1.mat");
  end
  r_bl = extract_subvector(r, nt, nlocal, bl_elems);
  if outer_iteration == 1
    test_mfem(-r_bl, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_step1bl.mat");
  end
  %r_global = pad_subvector(r_bl, nt, nlocal, bl_elems);
  %r_nonbl = r - r_global; % r_nonbl contains residual away from region of interest
  %rhs_bl = extract_subvector(rhs, nt, nlocal, bl_elems);
  %rhs_global = pad_subvector(rhs_bl, nt, nlocal, bl_elems);
  %rhs_nonbl = rhs - rhs_global; % rhs_nonbl contains RHS away from region of interest
  x_bl = extract_subvector(x, nt, nlocal, bl_elems);

  % Solving this problem "correctly" should ensure the outer iteration is
  % independent of the smallest bl element sizes. TODO: not quite clear what subtol should be
  subtol = 1e-3; % TODO: Pure subiteration
  %if outer_iteration > 4 % TODO: ramped subiteration
  %  subtol = tol*1e2;
  %else
  %  subtol = tol*1e-2;
  %end
  % TODO: Remove linear iterative solver and go back to GMRES

  if outer_iteration == 1
    test = precond_bl(r_bl);
    test_mfem(-test, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_precond.mat");
    
    test = A_bl(x_bl);
    test_mfem(test, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_Aloc.mat");
  end

  [e_bl, subiter, residuals] = restarted_fgmres(A_bl, r_bl, x_bl, subtol, nsubiter, nsubiter, precond_bl, true);
  %[e_bl, subiter, residuals] = static_gmres(A_bl, r_bl, x_bl, subtol, nsubiter, precond_bl, "right", false);
  if outer_iteration == 1
    test_mfem(-e_bl, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_step2.mat");
  end
  %iteration = init_jacobi_method(A_bl, diagA_bl, r_bl);
  %[e_bl,subiter,residuals] = iterative_method(A_bl, r_bl, x_bl, subtol, nsubiter, iteration, true);
  fprintf("Subiteration took %d iterations to achieve ||res|| = %8.2e\n", subiter, residuals(end));
  if subiter > 0
    x = x + pad_subvector(e_bl, nt, nlocal, bl_elems);
  end

  if outer_iteration == 1
    test_mfem(x, "/home/mfranco/scratch/2021/wr-les-solvers/results/temp/mult_step3.mat");
  end

  % TODO: Remove this check
  %if isa(A, 'function_handle')
  %  r = rhs - A(x);
  %else
  %  r = rhs - A*x;
  %end
  %fprintf("After preconditioner, ||r||=%8.2e\n", norm(r));
  
end

% Evaluates global preconditioner to rhs. In this case, just block diagonal inverse x = D\rhs
function x = apply_global_preconditioner(Dinvs, rhs)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  
  x = rhs;
  x2 = reshape(x, nlocal, nt);
  for it = 1:nt
    x2(:,it) = Dinvs{it}\x2(:,it);
  end
  x = reshape(x2, nlocal*nt, 1);
end

% Extract portion of operator A corresponding to rows specified by bl_elems
function [A_bl, diagA_bl] = extract_suboperator(Atimes, diagA, bl_elems)
  A_bl = @(x_bl) evaluate_suboperator(Atimes, diagA, bl_elems, x_bl);
  diagA_bl = diagA(:,:,bl_elems);
end

% Evaluates rhs_bl = A_bl*x_bl, when x is a subvector
function rhs_bl = evaluate_suboperator(Atimes, diagA, bl_elems, x_bl)
  nt = size(diagA,3);
  nlocal = size(diagA,2);

  x = pad_subvector(x_bl, nt, nlocal, bl_elems);
  rhs = Atimes(x);
  rhs_bl = extract_subvector(rhs, nt, nlocal, bl_elems);
end

% Given a global vector x, return the portion of the vector corresponding
% to the elements specified by bl_elems array (not a reference).
function x_bl = extract_subvector(x, nt, nlocal, bl_elems)
  x2 = reshape(x, nlocal, nt);
  x_bl2 = x2(:,bl_elems);
  x_bl = reshape(x_bl2, nlocal*length(bl_elems), 1);
end

% Given a subvector x_bl, return a padded, global vector where
% x=x_bl for all the elements specified by bl_elems, and x=pad_val for all others
function x = pad_subvector(x_bl, nt, nlocal, bl_elems, pad_val)
  if nargin < 5
    pad_val = 0;
  end

  x_bl_shape = reshape(x_bl, nlocal, length(bl_elems));
  if pad_val == 0
    x_shape = zeros(nlocal, nt);
  else
    x_shape = ones(nlocal,nt)*pad_val;
  end
  x_shape(:,bl_elems) = x_bl_shape(:,:);
  x = reshape(x_shape, nlocal*nt, 1);
end




% Test that matvec done in Matlab is same as in MFEM C++
function test_matvec(Atimes, b, mass_dir)
  Ab1 = Atimes(b);
  Ab2 = load(mass_dir+"benchmark_matvec.mat", "-ascii");

  norm(b)
  norm(Ab1)
  norm(Ab2)
  fprintf("diff = %8.3e\n", norm(Ab1-Ab2));
end

% Test that vector in Matlab is same as in MFEM C++
function test_mfem(v, mfem_file)
  v2 = load(mfem_file, "-ascii");

  fprintf("  Testing MFEM: %8.3e | %8.3e\n", norm(v), norm(v2));
  fprintf("  Testing MFEM: diff = %8.3e\n", norm(v-v2));
end
