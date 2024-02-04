export Tensor, matmul, backward, zero_grad!, zero_grad!!, relu, sigmoid

"""
    Tensor

A mutable structure designed to encapsulate data for automatic differentiation
within the `AutoDiff` module. It is a fundamental component for constructing
computational graphs, facilitating the computation of gradients necessary
for optimization algorithms, especially in machine learning contexts.

# Fields
- `data::AbstractArray{<:Real}`: Holds the tensor's numerical data.

- `parents::AbstractSet{Tensor}`: References the tensor's immediate predecessors within
a computational graph, essential for the backpropagation algorithm to compute gradients.

- `require_grad::Bool`: Indicates if the tensor should be considered for gradient
computation during backpropagation, enabling selective participation in the
computational graph based on the need for optimization.

- `grad::AbstractArray{Float32}`: Stores the computed gradient of the tensor as
a result of backpropagation, representing the partial derivatives of a target
function with respect to this tensor's data.

- `update!::Function`: A custom function assigned to update the tensor's gradient
during the backpropagation process, tailored to the specific operations that produced the
tensor.
"""
mutable struct Tensor
    data::AbstractArray{<:Real}
    label::AbstractString
    operation::AbstractString
    parents::AbstractSet{Tensor}
    require_grad::Bool
    grad::AbstractArray{Float32}
    update!::Function
end

function Tensor(
    data::AbstractArray{<:Real};
    label::AbstractString="",
    operation::AbstractString="+",
    parents::AbstractSet{Tensor}=Set{Tensor}(),
    require_grad::Bool=false,
)
    return Tensor(data, label, operation, parents, require_grad,
        zeros(Float32, size(data)), () -> return)
end

function Tensor(data::Real; kwargs...)
    return Tensor([data]; kwargs...)
end

function Base.show(io::IO, a::Tensor)
    print(io, "Tensor(")
    show(io, a.data)
    a.require_grad && print(io, ", require_grad=true")
    print(io, ")")
end

"""
    reshape_grad(grad::AbstractArray, target_shape::Tuple)

Reshapes the gradient `grad` to match a specified `target_shape`.
This function is useful in automatic differentiation to ensure that gradients
propagated back through operations match the shapes of the corresponding tensors.
The function handles different scenarios:

1. If `grad` already matches `target_shape`, it is returned as is.
2. If `target_shape` is an empty tuple `()`, indicating a scalar,
the sum of `grad` is returned.
3. Otherwise, it reshapes `grad` by summing over dimensions that do not match
`target_shape` and then reshaping the result to fit `target_shape`.

This operation is essential when gradients from operations involving broadcasting
or reduction need to be properly aligned with the original tensor shapes for
accurate gradient updates.

# Arguments
- `grad::AbstractArray`: The gradient array to be reshaped.
- `target_shape::Tuple`: The target shape to which `grad` should be reshaped.

# Returns
- `AbstractArray`: The reshaped gradient array.
"""
function reshape_grad(grad::AbstractArray, target_shape::Tuple)
    grad_shape = size(grad)
    grad_shape == target_shape && return grad
    target_shape == () && return sum(grad)

    shape = collect(target_shape)
    while length(shape) != length(grad_shape)
        push!(shape, 1)
    end

    dims = [i for i in 1:length(shape) if shape[i] == 1 && grad_shape[i] != 1]

    return reshape(sum(grad; dims), target_shape)
end

function Base.:+(a::Tensor, b::Tensor)
    parents = Set{Tensor}([a, b])
    require_grad = a.require_grad || b.require_grad
    out = Tensor(a.data .+ b.data; parents, require_grad, operation="+")
    out.update! = () -> begin
        a.require_grad && (a.grad += reshape_grad(out.grad, size(a.grad)))
        b.require_grad && (b.grad += reshape_grad(out.grad, size(b.grad)))
    end

    return out
end

function Base.:*(a::Tensor, b::Tensor)
    parents = Set{Tensor}([a, b])
    require_grad = a.require_grad || b.require_grad
    out = Tensor(a.data .* b.data; parents, require_grad, operation="*")
    out.update! = () -> begin
        a.require_grad && (a.grad += reshape_grad(out.grad .* b.data, size(a.grad)))
        b.require_grad && (b.grad += reshape_grad(out.grad .* a.data, size(b.grad)))
    end

    return out
end

function Base.:^(a::Tensor, b::Real)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(a.data .^ b; parents, require_grad, operation="^")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* b .* (a.data .^ (b - 1)))
    end

    return out
end

function Base.inv(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(inv.(a.data); parents, require_grad, operation="inv")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* (-1) .* (a.data .^ (-2)))
    end

    return out
end

function Base.sin(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(sin.(a.data); parents, require_grad, operation="sin")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* cos.(a.data))
    end

    return out
end

function Base.cos(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(cos.(a.data); parents, require_grad, operation="cos")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* (-1) .* sin.(a.data))
    end

    return out
end

function Base.exp(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(exp.(a.data); parents, require_grad, operation="exp")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* out.data)
    end

    return out
end

function Base.log(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(log.(a.data); parents, require_grad, operation="log")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad ./ a.data)
    end

    return out
end

function Base.transpose(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(transpose(a.data); parents, require_grad, operation="transpose")
    out.update! = () -> begin
        a.require_grad && (a.grad += transpose(out.grad))
    end

    return out
end

function matmul(a::Tensor, b::Tensor)
    parents = Set{Tensor}([a, b])
    require_grad = a.require_grad || b.require_grad
    out = Tensor(a.data * b.data; parents, require_grad, operation="matmul")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad * transpose(b.data))
        b.require_grad && (b.grad += transpose(a.data) * out.grad)
    end

    return out
end

relu(x) = max(0, x)

function relu(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(relu.(a.data); parents, require_grad, operation="ReLU")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* (a.data .>= 0))
    end

    return out
end

sigmoid(x) = 1 / (1 + exp(-x))

function sigmoid(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(sigmoid.(a.data); parents, require_grad, operation="sigmoid")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* out.data .* (1 .- out.data))
    end

    return out
end

function Base.tanh(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(tanh.(a.data); parents, require_grad, operation="tanh")
    out.update! = () -> begin
        a.require_grad && (a.grad += out.grad .* (1 .- (out.data .^ 2)))
    end

    return out
end

Base.:+(a::Tensor, b) = a + Tensor(b)
Base.:+(a, b::Tensor) = Tensor(a) + b
Base.:*(a::Tensor, b) = a * Tensor(b)
Base.:*(a, b::Tensor) = Tensor(a) * b
Base.:-(a::Tensor) = a * (-1)
Base.:-(a::Tensor, b::Tensor) = a + (-b)
Base.:-(a, b::Tensor) = Tensor(a) - b
Base.:-(a::Tensor, b) = a - Tensor(b)
Base.:/(a::Tensor, b::Tensor) = a * inv(b)
Base.:/(a, b::Tensor) = Tensor(a) / b
Base.:/(a::Tensor, b) = a / Tensor(b)
matmul(a, b::Tensor) = matmul(Tensor(a), b)
matmul(a::Tensor, b) = matmul(a, Tensor(b))

"""
    zero_grad!(a::Tensor)

Resets the gradient of tensor `a` to zero. This is particularly useful in iterative
optimization algorithms where gradients are accumulated, and need to be reset
after each update step.

# Arguments
- `a::Tensor`: The tensor for which to reset the gradient.
"""
function zero_grad!(a::Tensor)
    a.grad = zeros(Float32, size(a.grad))
end

"""
    zero_grad!!(a::Tensor)

Recursively resets the gradients of all tensors in the computational graph to which
tensor `a` belongs, to zero. This function traverses the entire computational graph
starting from `a`, ensuring that every tensor's gradient is reset. This is useful
for resetting gradients of all parameters in a neural network before the next
forward and backward pass.

# Arguments
- `a::Tensor`: The tensor from which to start resetting gradients.
This is typically an output tensor of a computational graph.
"""
function zero_grad!!(a::Tensor)
    nodes = topological_sort(a)
    for node in nodes
        zero_grad!(node)
    end
end

"""
    topological_sort(a::Tensor) -> Vector{Tensor}

Performs a topological sort on the computational graph starting from tensor `a`.
This function identifies the order in which operations must be executed to correctly
perform the backward pass of automatic differentiation.

# Arguments
- `a::Tensor`: The tensor from which to start the topological sort.

# Returns
- `Vector{Tensor}`: A vector of tensors sorted in topological order.
"""
function topological_sort(a::Tensor)
    nodes = Vector{Tensor}()
    visited = Set{Tensor}()

    function build(node::Tensor)
        if !(node ∈ visited)
            push!(visited, node)
            for parent in node.parents
                build(parent)
            end

            push!(nodes, node)
        end
    end

    build(a)

    return nodes
end

"""
    backward(a::Tensor)

Performs the backpropagation algorithm starting from tensor `a`. This function computes
the gradients of `a` with respect to all tensors in its computational graph that have
`require_grad` set to `true`. It operates in two steps:
1. Calls `topological_sort()` to order the tensors in a way that respects their
computational dependencies.
2. Iteratively applies the stored `update!` functions in reverse topological order to
propagate gradients back through the graph.

# Arguments
- `a::Tensor`: The tensor from which to start backpropagation. It is assumed that this
tensor is the output of some computation, typically a loss function in machine
learning contexts.
"""
function backward(a::Tensor)
    nodes = topological_sort(a)
    a.grad = ones(Float32, size(a.grad))
    for node in reverse(nodes)
        node.update!()
    end
end
