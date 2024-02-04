export show_graph

import Graphs: DiGraph, add_edge!, add_vertex!
import GraphRecipes: graphplot
import Plots: plot, savefig

"""
    show_graph(root::Tensor; kwargs...)

Plots the computational graph starting from the specified `root` tensor.
This function traverses the computational graph to collect all unique tensors and their
connections, then visualizes the graph, showing the flow of operations and gradients.
This recipe can be shown using graphplot(a::Tensor) from Plots library.

# Arguments
- `root::Tensor`: The root tensor from which to start plotting the computational graph.
"""
function show_graph(root::Tensor; save_path=nothing, nodeshape=:circle, curves=false,
    node_size=0.2, fontsize=10, nodecolor=:lightgray, kwargs...)

    nodes = topological_sort(root)
    tensor2index = Dict{Tensor,Int}()
    for (index, node) in enumerate(nodes)
        tensor2index[node] = index
    end

    n_nodes = length(nodes)
    g = DiGraph(n_nodes)
    names = Vector{String}(undef, n_nodes)

    n_operations = 0
    n_unnamed = 0
    for (index, node) in enumerate(nodes)
        if isempty(node.label)
            n_unnamed += 1
            names[index] = "t$n_unnamed"
        else
            names[index] = node.label
        end

        isempty(node.parents) && continue

        n_operations += 1
        operation_index = n_nodes + n_operations
        add_vertex!(g)
        push!(names, node.operation)
        add_edge!(g, operation_index, index)

        for parent in node.parents
            add_edge!(g, tensor2index[parent], operation_index)
        end
    end

    p = graphplot(g; names, nodeshape, curves, node_size, fontsize, nodecolor, kwargs...)
    save_path !== nothing ? savefig(p, save_path) : plot(p)
end
