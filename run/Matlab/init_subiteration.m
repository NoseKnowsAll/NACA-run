% Creates a subiteration preconditioner to solve A\b
function precond = init_subiteration(A, diagA, Ms, bl_elems, b, global_precond_type, tol, nsubiter, subtol_factor)
  nt = size(diagA,3);
  nlocal = size(diagA,2);
  fprintf("Subiteration preconditioner with nsubiter=%d, nt=%d, size(bl_elems)=%d\n", nsubiter, nt, length(bl_elems));

  if global_precond_type == "jacobi"
    precond_global = init_jacobi(diagA);
  elseif global_precond_type == "mass_inv"
    precond_global = init_jacobi(Ms);
  else
    error("Invalid global_precond_type");
  end

  [A_bl, diagA_bl] = extract_suboperator(A, diagA, bl_elems);
  
  precond_bl = init_jacobi(diagA_bl);
  precond = @(rhs) evaluate_subiteration(A, A_bl, diagA_bl, precond_global, precond_bl, tol, nsubiter, subtol_factor, bl_elems, nt, nlocal, rhs);
  
  global outer_iteration;
  global inner_iterations;
  global percent_subregion;
  outer_iteration = 0;
  inner_iterations = 0;
  percent_subregion = length(bl_elems)/nt;
end

% Subiteration preconditioner is one application of x = P\rhs.
% First apply block diagonal inverse to entire region. x = D\rhs
% Then consider just the boundary layer elements specified by bl_elems.
% Compute boundary layer residual r_{bl} = rhs_{bl} - A_{bl}*x_{bl}
% Perform inner GMRES iteration to solve A_{bl}*e_{bl} = r_{bl} with initial guess x_{bl} to get e_{bl}
% Update x according to correction: x = x + e_{bl}
% Note: Only valid as a preconditioner within a broader FGMRES iteration
function x = evaluate_subiteration(A, A_bl, diagA_bl, precond_global, precond_bl, tol, nsubiter, subtol_factor, bl_elems, nt, nlocal, rhs)

  x = precond_global(rhs);
  
  global outer_iteration;
  outer_iteration = outer_iteration + 1;
  if nsubiter <= 0
    return;
  end

  % TODO: Debugging compute_subtol
  %[x_true, iter, res] = static_gmres(A, rhs, x, 1e-10, 300, precond_global, "right", false);
  %error = x_true - x;
  %iter = num2str(outer_iteration, "%03.f");
  %err_file = sprintf("../results/Matlab/error_it%s.mat", iter);
  %fwritearray(err_file, error);
  
  % TODO: check if convergence changes when we compute true r_bl = rhs_bl - A_bl(x_bl)
  if isa(A, 'function_handle')
    r = rhs - A(x);
  else
    r = rhs - A*x;
  end

  %TODO: Debugging
  %scaled_err = precond_global(r);
  %iter = num2str(outer_iteration, "%03.f");
  %res_file = sprintf("../results/Matlab/residual_it%s.mat", iter);
  %fwritearray(res_file, r);
  
  %scaled_err_file = sprintf("../results/Matlab/scaled_error_it%s.mat", iter);
  %fwritearray(scaled_err_file, scaled_err);
  
  r_bl = extract_subvector(r, nt, nlocal, bl_elems);

  %r_global = pad_subvector(r_bl, nt, nlocal, bl_elems);
  %r_nonbl = r - r_global; % r_nonbl contains residual away from region of interest
  %rhs_bl = extract_subvector(rhs, nt, nlocal, bl_elems);
  %rhs_global = pad_subvector(rhs_bl, nt, nlocal, bl_elems);
  %rhs_nonbl = rhs - rhs_global; % rhs_nonbl contains RHS away from region of interest
  x_bl = extract_subvector(x, nt, nlocal, bl_elems);

  % Solving this problem "correctly" should ensure the outer iteration is
  % independent of the smallest bl element sizes.

  % TODO: Debugging compute_subtol
  %subtol = compute_subtol_from_scaled_error(r, precond_global, nt, nlocal, bl_elems)*subtol_factor;
  scaled_err = precond_global(r);
  subtol = compute_subtol_from_error(scaled_err, nt, nlocal, bl_elems)*subtol_factor;
  %subtol = compute_subtol(r, nt, nlocal, bl_elems)*subtol_factor;
  fprintf("subtol*factor computed to be %8.3e\n", subtol);
  if subtol == 0
    return;
  end

  % Initial guess -x_bl simply to align with MFEM solving negative of this problem
  %[e_bl, subiter, residuals] = restarted_fgmres(A_bl, r_bl, -x_bl, subtol, nsubiter, nsubiter, precond_bl, false);
  %[e_bl, subiter, residuals] = static_gmres(A_bl, r_bl, -x_bl, subtol, nsubiter, precond_bl, "right", false);
  [e_bl, subiter, residuals] = static_gmres(A_bl, r_bl, -x_bl, subtol, nsubiter, precond_bl, "left", false);

  %iteration = init_jacobi_method(A_bl, diagA_bl, r_bl);
  %[e_bl,subiter,residuals] = iterative_method(A_bl, r_bl, x_bl, subtol, nsubiter, iteration, true);
  fprintf("Subiteration took %d iterations to achieve ||res|| = %8.2e\n", subiter, residuals(end));
  if subiter > 0
    global inner_iterations;
    inner_iterations = inner_iterations + subiter;
    x = x + pad_subvector(e_bl, nt, nlocal, bl_elems);
  end
  
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
  
  %nelem_worst = 10; % Number of worst elements to consider in both subregion and outer region
  %least_improvement_bls    = mink(norms(bl_elems)   , nelem_worst);
  %least_improvement_nonbls = mink(norms(nonbl_elems), nelem_worst);
  %least_improvement_bl     = exp(mean(log(least_improvement_bls))); % log mean instead so focus on order of magnitudes
  %least_improvement_nonbl  = exp(mean(log(least_improvement_nonbls)));
  fprintf("Least improvement found in bl: %8.3e\n", least_improvement_bl);
  fprintf("Least improvement found outside bl: %8.3e\n", least_improvement_nonbl);
  if least_improvement_bl >= least_improvement_nonbl
    subtol = 0; % skip inner GMRES completely because we've improved more in subregion
  else
    subtol = least_improvement_bl/least_improvement_nonbl;
  end
end


% Compute the subtolerance needed to solve inner problem.
% After solve, this tolerance ensures element with least improvement in subregion
% will be at least as accurate as element with least improvement in rest of domain.
% error = true error = A\rhs - P\rhs
function subtol = compute_subtol_from_error(error, nt, nlocal, bl_elems)
  error2 = reshape(error, nlocal, nt);
  norms = vecnorm(error2);
  nonbl_elems = setdiff(1:nt, bl_elems);

  max_error_bl = max(norms(bl_elems));
  max_error_nonbl = max(norms(nonbl_elems));
  fprintf("max error found in bl: %8.3e\n", max_error_bl);
  fprintf("max error found outside bl: %8.3e\n", max_error_nonbl);
  if max_error_bl < max_error_nonbl
    subtol = 0; % skip inner GMRES completely because we've improved more in subregion
  else
    subtol = max_error_nonbl/max_error_bl;
  end
end

% Compute the subtolerance needed to solve inner problem.
% scaled_error = P\res which should approximately be error, element scaling-wise
% Then, multiply by ||res||/||scaled_error|| to convert tolerance to be wrt res
% subtol: Dimensionless amount residual must improve
function subtol = compute_subtol_from_scaled_error(r, precond_global, nt, nlocal, bl_elems)
  scaled_error = precond_global(r);
  scaled_error2 = reshape(scaled_error, nlocal, nt);
  r2 = reshape(r, nlocal, nt);
  norms = vecnorm(scaled_error2);
  nonbl_elems = setdiff(1:nt, bl_elems);

  [max_error_bl, ibl] = max(norms(bl_elems));
  [max_error_nonbl, inbl] = max(norms(nonbl_elems));
  fprintf("max error found in bl: %8.3e\n", max_error_bl);
  fprintf("max error found outside bl: %8.3e\n", max_error_nonbl);
  if max_error_bl < max_error_nonbl
    subtol = 0; % skip inner GMRES because subregion already more accurate
  else
    err2res_bl = norm(r2(:,bl_elems(ibl)))/max_error_bl;
    err2res_nbl = norm(r2(:,nonbl_elems(inbl)))/max_error_nonbl;
    fprintf("err2res_bl = %8.3e, err2res_nbl = %8.3e\n", err2res_bl, err2res_nbl);
    subtol = (max_error_nonbl*err2res_nbl) / (max_error_bl*err2res_bl);
    if subtol > 1
      subtol = 0; % skip inner GMRES because subregion already more accurate (relatively)
    end
  end
end
