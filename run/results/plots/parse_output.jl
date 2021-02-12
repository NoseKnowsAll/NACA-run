function read_numbers(filename)
    istage = 1
    istep = 1
    convergence_rates = Array{Array{Array{Float64,1},1},1}()
    assemble_times    = Array{Array{Array{Float64,1},1},1}()
    solve_times       = Array{Array{Array{Float64,1},1},1}()
    krylov_iters      = Array{Array{Array{Int,1},1},1}()
    float_reg = "([+-]?[0-9]*[.]?[0-9]+)"
    for (i,line) in enumerate(readlines(filename))
        if length(line) < 2
            continue
        elseif line[1:4] == "DIRK" # New stage
            istage = parse(Int, line[end-2]) # Assumes single digit number of stages!
            if istage > 1
                push!(convergence_rates[istep], Float64[])
                push!(assemble_times[istep], Float64[])
                push!(solve_times[istep], Float64[])
                push!(krylov_iters[istep], Int[])
            end
        elseif line[1:4] == "===="
            continue
        elseif line[1:4] == "  It"
            continue
        elseif line[1:4] == " >>>" # New time step
            m = match(r"(\d+)", line)
            istep = parse(Int, m.captures[1])
            push!(convergence_rates, [Float64[]])
            push!(assemble_times, [Float64[]])
            push!(solve_times, [Float64[]])
            push!(krylov_iters, [Int[]])
        else # New row in table of information
            m = match(r"(\d+)\s+([+-]?[0-9]*[.]?[0-9]+)\s+([+-]?[0-9]*[.]?[0-9]+)\s+([+-]?[0-9]*[.]?[0-9]+)\s+(\d+)", line)
            if isnothing(m)
                m = match(r"(\d+)\s+([+-]?[0-9]*[.]?[0-9]+)\s+([+-]?[0-9]*[.]?[0-9]+)", line)
            end
            push!(convergence_rates[istep][istage], parse(Float64,m.captures[2]))
            if length(m.captures) > 3
                # Assemble times are meaningless when Jacobian is not actually being assembled
                push!(assemble_times[istep][istage], parse(Float64, m.captures[3]))
                push!(solve_times[istep][istage], parse(Float64, m.captures[4]))
                push!(krylov_iters[istep][istage], parse(Int, m.captures[5]))
            end
        end
    end
    return convergence_rates, assemble_times, solve_times, krylov_iters
end

function combine_results(convergence_rates, assemble_times, solve_times, krylov_iters)
    nested_size(nested_array) = maximum([length(nested_array[i][j]) for i = 1:length(nested_array) for j = 1:length(nested_array[i])])
    convergence_rates2 = zeros(nested_size(convergence_rates))
    assemble_times2 = zeros(nested_size(assemble_times))
    solve_times2 = zeros(nested_size(solve_times))
    krylov_iters2 = zeros(nested_size(krylov_iters)) # average is not an Int

    # e^ { mean(log(rates)) }
    counts = zeros(Int, nested_size(convergence_rates))
    for i = 1:length(convergence_rates)
        for j = 1:length(convergence_rates[i])
            for k = 1:length(convergence_rates[i][j])
                counts[k] += 1
                convergence_rates2[k] += log(convergence_rates[i][j][k])
            end
        end
    end
    convergence_rates2 .= exp.(convergence_rates2./counts)

    # Mean of the rest
    counts = zeros(Int, nested_size(assemble_times)) # Steps,stages being averaged over
    for i = 1:length(assemble_times)
        for j = 1:length(assemble_times[i])
            for k = 1:length(assemble_times[i][j])
                counts[k] += 1
                assemble_times2[k] += assemble_times[i][j][k]
                solve_times2[k] += solve_times[i][j][k]
                krylov_iters2[k] += krylov_iters[i][j][k]
            end
        end
    end
    assemble_times2 ./= counts
    solve_times2 ./= counts
    krylov_iters2 ./= counts

    return convergence_rates2, assemble_times2, solve_times2, krylov_iters2
end

"""
    convergence_rates, assemble_times, solve_times, krylov_iters = parse_output(filename)
Outward facing function: Parse a given output.txt file for convergence rates, 
assembly times, solve times, and krylov iterations. Average out across all steps and stages.
Convergence rates are averaged according to: exp( mean(log(rates)) )
"""
function parse_output(filename)
    values = read_numbers(filename)
    convergence_rates, assemble_times, solve_times, krylov_iters = combine_results(values...)
end
