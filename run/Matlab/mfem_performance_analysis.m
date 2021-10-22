% Check performance of subiteration solver on MFEM convection-diffusion problem to analyze several different hyperparameters
function mfem_performance_analysis(run_tests, plot_tests, results_file)

  % Initialization
  if nargin < 3
    results_file = "/scratch/mfranco/2021/naca/run/results/Matlab/mfem_performance.mat";
    if nargin < 2
      plot_tests = false;
      if nargin < 1
	run_tests = false;
      end
    end
  end

  if run_tests
    % Run tests - expensive
    global_precond_type = "jacobi";
    tol = 1e-8;
    dts = [1e-3];
    Lxs = 1:10;
    subtol_factors = [1e0 1e1];
    nsubiters = [10 20 30 40 300];
    performance = zeros(length(nsubiters), length(subtol_factors), length(dts), 4);

    for iLx = 1:length(Lxs)
      Lx = Lxs(iLx);
      for idt = 1:length(dts)
	dt = dts(idt);

	for ist = 1:length(subtol_factors)
	  subtol_factor = subtol_factors(ist);

	  for ins = 1:length(nsubiters)
	    nsubiter = nsubiters(ins);

	    mfem_subiteration_driver(dt, tol, nsubiter, subtol_factor, global_precond_type, Lx);
	    
	    global inner_iterations;
	    global outer_iteration;
	    global percent_subregion;
	    global fgmres_timer;
	    performance(ins, ist, idt, iLx, 1) = inner_iterations;
	    performance(ins, ist, idt, iLx, 2) = outer_iteration;
	    performance(ins, ist, idt, iLx, 3) = cost_in_matvecs(inner_iterations, outer_iteration, percent_subregion);
	    performance(ins, ist, idt, iLx, 4) = fgmres_timer;
	    fprintf("\n\n\n\n");
	    
	  end
	end
      end
    end

    disp(performance);
    save(results_file, "performance", "dts", "subtol_factors", "nsubiters");
  else
    % Load test results instead: dts, subtol_factors, nsubiters
    load(results_file);
  end

  % Plot tests if interested
  if plot_tests
    for idt = 1:length(dts);
      dt = dts(idt);

      for ist = 1:length(subtol_factors)
	subtol_factor = subtol_factors(ist);

	% Plot cost vs nsubiters for this (dt, subtol_factor) pair
	figure((idt-1)*length(subtol_factors)+ist);
	plot(nsubiters, performance(:,ist,idt,3), 'r.-', 'markersize', 20);
	title(sprintf("dt=%.1e, subtolFactor=%.1e",dt, subtol_factor));
	xlabel("nsubiter");
	ylabel("effective matvecs");
      end
    end
  end

  % Print results at the end
  for idt = 1:length(dts)
    dt = dts(idt);
    for ist = 1:length(subtol_factors)
      subtol_factor = subtol_factors(ist);
      for ins = 1:length(nsubiters)
	nsubiter = nsubiters(ins);

	fprintf("@ dt=%.1e, subtol_factor=%.1e, nsubiter=%3d, (inner, outer, cost) = (%d, %d, %f)\n", dt, subtol_factor, nsubiter, performance(ins,ist,idt,1), performance(ins,ist,idt,2), performance(ins,ist,idt,3));
      end
    end
  end

end

% Computes the cost of a subiteration solve in terms of global matvecs
% cost = n_element*n_local^2 * (3*it_outer + 2*percent_subregion*it_inner)
% it_outer is the global number of outer iterations
% it_inner is the total number of inner iterations across all outer iterations
% percent subregion = n_elements / n_elements_in_subregion
function cost = cost_in_matvecs(it_inner, it_outer, percent_subregion)
  cost = 3*it_outer + 2*percent_subregion*it_inner;
end
