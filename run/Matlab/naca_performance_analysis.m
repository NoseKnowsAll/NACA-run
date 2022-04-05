% Check performance of subiteration solver on NACA problem to analyze several different hyperparameters
function naca_performance_analysis(run_tests, plot_tests, msh_name, results_file)

  % Initialization
  if nargin < 3; msh_name = "naca_v2_p3_r12"; end;
  if nargin < 2; plot_tests = false; end;
  if nargin < 1; run_tests = false; end;
  if nargin < 4
    results_file = "/scratch/mfranco/2021/naca/run/results/Matlab/"+msh_name+"_performance.mat";
  end

  if run_tests
    % Run tests - expensive
    global_pre = "jacobi";
    inner_pre = "ilu";
    tol = 1e-8;
    dts = [1e-3];
    subtol_factors = [1.0];
    nsubiters = [0 20 40 100];
    performance = zeros(length(nsubiters), length(subtol_factors), length(dts), 4);
    
    for idt = 1:length(dts);
      dt = dts(idt);

      for ist = 1:length(subtol_factors)
	subtol_factor = subtol_factors(ist);

	for ins = 1:length(nsubiters);
	  nsubiter = nsubiters(ins);
	
	  naca_subiteration_driver(dt, tol, nsubiter, subtol_factor, global_pre, inner_pre, msh_name);
	
	  global inner_iterations;
	  global outer_iteration;
	  global percent_subregion;
	  global fgmres_timer;
	  performance(ins, ist, idt, 1) = inner_iterations;
	  performance(ins, ist, idt, 2) = outer_iteration;
	  performance(ins, ist, idt, 3) = cost_in_matvecs(inner_iterations, outer_iteration, percent_subregion);
	  performance(ins, ist, idt, 4) = fgmres_timer;
	  fprintf("\n\n\n\n");
	  
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
    close all;
    for idt = 1:length(dts);
      dt = dts(idt);

      for ist = 1:length(subtol_factors)
	subtol_factor = subtol_factors(ist);

	% Plot cost vs nsubiters for this (dt, subtol_factor) pair
	figure((idt-1)*length(subtol_factors)+ist);
	yyaxis left;
	ylabel("matvecs");
	plot(nsubiters, performance(:,ist,idt,3), 'r.-', 'markersize', 20);
	hold on;
	timings = performance(:,ist,idt,4);
	yyaxis right;
	ylabel("timing (s)");
	plot(nsubiters, timings, 'b--', 'markersize', 15);
	title(sprintf("%s dt=%.1e, subtolFactor=%.1e", msh_name, dt, subtol_factor), 'interpreter', 'none');
	xlabel("nsubiter");
	legend("performance model", "scaled timings");
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

	fprintf("@ dt=%.1e, subtol_factor=%.1e, nsubiter=%3d, (inner, outer, cost) = (%d, %d, %f), t=%.2e\n", dt, subtol_factor, nsubiter, performance(ins,ist,idt,1), performance(ins,ist,idt,2), performance(ins,ist,idt,3), performance(ins,ist,idt,4));
      end
    end
  end

end

% Computes the cost of a subiteration solve in terms of global matvecs
% cost = n_element*n_local^2 * ((2+1/4)*it_outer + (1+1)*percent_subregion*it_inner)
% it_outer is the global number of outer iterations
% it_inner is the total number of inner iterations across all outer iterations
% percent subregion = n_elements / n_elements_in_subregion
function cost = cost_in_matvecs(it_inner, it_outer, percent_subregion)
  % OLD COMPUTATION
  %cost = 3*it_outer + 2*percent_subregion*it_inner;
  % One less matvec in outer iteration because we now directly use residual, not preconditioned residual
  cost = (2+1/4)*it_outer + (1+1)*percent_subregion*it_inner;
end
