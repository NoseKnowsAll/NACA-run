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
  
  global outer_iteration;
  outer_iteration = outer_iteration + 1;
  %if outer_iteration <= 6 || outer_iteration > 8 % TODO: ramped subiteration
  %  nsubiter = 0;
  %end
  %nsubiter = 0; % TODO: only global preconditioner
  if nsubiter <= 0
    return;
  end
  
  % TODO: check if convergence changes when we compute true r_bl = rhs_bl - A_bl(x_bl)
  if isa(A, 'function_handle')
    r = rhs - A(x);
  else
    r = rhs - A*x;
  end

  %iter = num2str(outer_iteration, "%03.f");
  %res_file = sprintf("../results/Matlab/residual_it%s.mat", iter);
  %fwritearray(res_file, r);
  r_bl = extract_subvector(r, nt, nlocal, bl_elems);

  %r_global = pad_subvector(r_bl, nt, nlocal, bl_elems);
  %r_nonbl = r - r_global; % r_nonbl contains residual away from region of interest
  %rhs_bl = extract_subvector(rhs, nt, nlocal, bl_elems);
  %rhs_global = pad_subvector(rhs_bl, nt, nlocal, bl_elems);
  %rhs_nonbl = rhs - rhs_global; % rhs_nonbl contains RHS away from region of interest
  x_bl = extract_subvector(x, nt, nlocal, bl_elems);

  % Solving this problem "correctly" should ensure the outer iteration is
  % independent of the smallest bl element sizes. TODO: not quite clear what subtol should be

  subtol = compute_subtol(r, nt, nlocal, bl_elems);
  fprintf("subtol computed to be %8.3e\n", subtol);
  if subtol == 0
    return
  end
  %subtol = 1e-1; % TODO: Pure subiteration
  %if outer_iteration > 4 % TODO: ramped subiteration
  %  subtol = tol;
  %else
  %  subtol = tol*1e-4;
  %end

  % Initial guess -x_bl simply to align with MFEM solving negative of this problem
  %[e_bl, subiter, residuals] = restarted_fgmres(A_bl, r_bl, -x_bl, subtol, nsubiter, nsubiter, precond_bl, false);
  [e_bl, subiter, residuals] = static_gmres(A_bl, r_bl, -x_bl, subtol, nsubiter, precond_bl, "right", false);

  %iteration = init_jacobi_method(A_bl, diagA_bl, r_bl);
  %[e_bl,subiter,residuals] = iterative_method(A_bl, r_bl, x_bl, subtol, nsubiter, iteration, true);
  fprintf("Subiteration took %d iterations to achieve ||res|| = %8.2e\n", subiter, residuals(end));
  if subiter > 0
    x = x + pad_subvector(e_bl, nt, nlocal, bl_elems);
  end
  
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

% Compute the subtolerance needed to solve inner problem.
% After solve, this tolerance ensures element with least improvement in subregion
% will be at least as accurate as element with least improvement in rest of domain.
function subtol = compute_subtol(r, nt, nlocal, bl_elems)
  r2 = reshape(r, nlocal, nt);
  norms = vecnorm(r2);
  nonbl_elems = setdiff(1:nt, bl_elems);
  least_improvement_bl = min(norms(bl_elems));
  least_improvement_nonbl = min(norms(nonbl_elems));
  fprintf("Least improvement found in bl: %8.3e\n", least_improvement_bl);
  fprintf("Least improvement found outside bl: %8.3e\n", least_improvement_nonbl);
  if least_improvement_nonbl < least_improvement_bl
    subtol = 0; % skip GMRES completely
  else
    subtol = least_improvement_bl/least_improvement_nonbl;
  end
end
