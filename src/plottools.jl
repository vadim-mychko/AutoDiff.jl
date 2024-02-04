import Graphs: DiGraph, add_edge!
import GraphRecipes: graphplot
import Plots: plot

"""
    show_graph(root::Tensor; kwargs...)

Plots the computational graph starting from the specified `root` tensor.
This function traverses the computational graph to collect all unique tensors and their
connections, then visualizes the graph, showing the flow of operations and gradients.
This recipe can be shown using graphplot(a::Tensor) from Plots library.

# Arguments
- `root::Tensor`: The root tensor from which to start plotting the computational graph.
"""
function show_graph(root::Tensor; nodeshape=:circle, curves=false, node_size=0.3,
    fontsize=10, nodecolor=:lightgray, method=:stress, kwargs...)

    nodes = topological_sort(root)
    tensor2id = Dict{Tensor,Int}()
    for (index, node) in enumerate(nodes)
        tensor2id[node] = index
    end

    n_nodes = length(nodes)
    g = DiGraph(n_nodes)
    names = Vector{String}(undef, n_nodes)
    edgelabel = Dict{Tuple{Int,Int},String}()

    for (index, node) in enumerate(nodes)
        names[index] = node.label
        for parent in node.parents
            parent_index = tensor2id[parent]
            add_edge!(g, parent_index, index)
            edgelabel[(parent_index, index)] = node.operation
        end
    end

    graphplot(g; names, edgelabel, nodeshape, curves, node_size, fontsize,
        nodecolor, method, kwargs...)
end
