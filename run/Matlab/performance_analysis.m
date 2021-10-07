% Check performance of subiteration solver on NACA problem to analyze several different hyperparameters
function performance_analysis(run_tests)

  if nargin < 1
    run_tests = false;
  end
  
  % Initialization
  dts = [1e-4, 1e-3, 1e-2];
  p   = 3;
  global_precond_type = "jacobi";
  tol = 1e-8;
  subtol_fudges = [1e0 1e1 1e2];
  nsubiters = [10 20 300];

  performance = zeros(length(nsubiters), length(subtol_fudges), length(dts), 3);

  if run_tests
    % Run tests 
    for idt = 1:length(dts);
      dt = dts(idt);

      for ist = 1:length(subtol_fudges)
	subtol_fudge = subtol_fudges(ist);

	for ins = 1:length(nsubiters);
	  nsubiter = nsubiters(ins);
	
	  naca_subiteration_driver(dt, tol, nsubiter, subtol_fudge, global_precond_type);
	
	  global inner_iterations;
	  global outer_iteration;
	  global percent_subregion;
	  performance(ins, ist, idt, 1) = inner_iterations;
	  performance(ins, ist, idt, 2) = outer_iteration;
	  performance(ins, ist, idt, 3) = cost_in_matvecs(inner_iterations, outer_iteration, percent_subregion);
	  fprintf("\n\n\n\n");
	  
	end
      end
    end

    disp(performance);
    save("../results/Matlab/performance.mat", "performance");
  else
    load("../results/Matlab/performance.mat");
  end

  % Print results at the end
  for idt = 1:length(dts)
    dt = dts(idt);
    for ist = 1:length(subtol_fudges)
      subtol_fudge = subtol_fudges(ist);
      for ins = 1:length(nsubiters)
	nsubiter = nsubiters(ins);

	fprintf("@ dt=%.1e, subtol_factor=%.1e, nsubiter=%3d, (inner, outer, cost) = (%d, %d, %f)\n", dt, subtol_fudge, nsubiter, performance(ins,ist,idt,1), performance(ins,ist,idt,2), performance(ins,ist,idt,3));
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
