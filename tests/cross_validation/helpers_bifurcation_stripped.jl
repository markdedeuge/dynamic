# Bifurcation helpers — stripped of Plots.jl dependency
# Extracted from reference/SCYFI/src/utilities/helpers_bifurcation.jl

"""
Compare the stability of one cycle to all neighbouring cycles
"""
function compare_stability(eigvals::Array, eigvals_neighbour::Array)
    same_stability_index = []
    norm_1 = abs.(eigvals).<1
    number_stable_dimensions_1 = sum(norm_1)
    for i in eachindex(eigvals_neighbour)
        norm_2 = abs.(eigvals_neighbour[i]).<1
        number_stable_dimensions_2 = sum(norm_2)
        if number_stable_dimensions_1==number_stable_dimensions_2
            append!(same_stability_index,i)
        end
    end
    if same_stability_index==[]
        return nothing
    end
    return same_stability_index
end

"""
Compare one cycle to all neighboring cycles and return the sum of the minimal euclidean distance
"""
function get_combined_state_space_eigenvalue_distance(cycle::Array, cycles_neighbour::Array,eigenvalue::Array, eigenvalue_neighbour::Array)
    distance_cycles = get_minimal_state_space_distances(cycle, cycles_neighbour)
    distance_eigvals = get_minimal_eigenvalue_distances(eigenvalue, eigenvalue_neighbour)
    return distance_cycles .+ distance_eigvals
end

"""
Compare one cycle to all neighboring cycles and return minimal distance in state space
"""
function get_minimal_state_space_distances(cycle::Array, cycles_neighbour::Array)
    distance_list = []
    for i in eachindex(cycles_neighbour)
        distance = Inf   
        for j in eachindex(cycles_neighbour[i])
            mean_distance = norm(cycle - circshift(cycles_neighbour[i],j))
            if mean_distance < distance
                distance = mean_distance
            end
        end
        append!(distance_list,distance)
    end
    return distance_list
end

"""
Compare one cycles eigenvalues to all neighboring cycles eigenvalues
"""
function get_minimal_eigenvalue_distances(eigenvalue::Array, eigenvalue_neighbour::Array)
    distance_list = []
    for i in eachindex(eigenvalue_neighbour)
        distance = Inf   
        for j in eachindex(eigenvalue_neighbour[i])
            mean_distance = norm(eigenvalue - circshift(eigenvalue_neighbour[i],j))
            if mean_distance < distance
                distance = mean_distance
            end
        end
        append!(distance_list,distance)
    end
    return distance_list
end
