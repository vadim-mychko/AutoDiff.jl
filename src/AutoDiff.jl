module AutoDiff

export Tensor, matmul, backward, zero_grad!, zero_grad!!

mutable struct Tensor
    data::AbstractArray{<:Real}
    parents::AbstractSet{Tensor}
    require_grad::Bool
    grad::AbstractArray{Float32}
    update!::Function
end

function Tensor(
    data::AbstractArray{<:Real};
    parents::AbstractSet{Tensor} = Set{Tensor}(),
    require_grad::Bool = false,
)
    return Tensor(data, parents, require_grad, zeros(Float32, size(data)), () -> return)
end

function Tensor(data::Real; kwargs...)
    return Tensor([data]; kwargs...)
end

function Base.show(io::IO, obj::Tensor)
    print(io, "Tensor(")
    show(io, obj.data)
    obj.require_grad && print(io, ", require_grad=true")
    print(io, ")")
end

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
    out = Tensor(a.data .+ b.data; parents, require_grad)
    out.update! = () -> begin
        a.grad += reshape_grad(out.grad, size(a.grad))
        b.grad += reshape_grad(out.grad, size(b.grad))
    end

    return out
end

function Base.:*(a::Tensor, b::Tensor)
    parents = Set{Tensor}([a, b])
    require_grad = a.require_grad || b.require_grad
    out = Tensor(a.data .* b.data; parents, require_grad)
    out.update! = () -> begin
        a.grad += reshape_grad(out.grad .* b.data, size(a.grad))
        b.grad += reshape_grad(out.grad .* a.data, size(b.grad))
    end

    return out
end

function Base.:^(a::Tensor, b::Real)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(a.data .^ b; parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad .* b .* (a.data .^ (b - 1))
    end

    return out
end

function Base.inv(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(inv.(a.data); parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad .* (-1) .* (a.data .^ (-2))
    end

    return out
end

function matmul(a::Tensor, b::Tensor)
    parents = Set{Tensor}([a, b])
    require_grad = a.require_grad || b.require_grad
    out = Tensor(a.data * b.data; parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad * transpose(b.data)
        b.grad += transpose(a.data) * out.grad
    end

    return out
end

function Base.sin(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(sin.(a.data); parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad .* cos.(a.data)
    end

    return out
end

function Base.cos(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(cos.(a.data); parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad .* (-1) .* sin.(a.data)
    end

    return out
end

function Base.exp(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(exp.(a.data); parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad .* out.data
    end

    return out
end

function Base.log(a::Tensor)
    parents = Set{Tensor}([a])
    require_grad = a.require_grad
    out = Tensor(log.(a.data); parents, require_grad)
    out.update! = () -> begin
        a.grad += out.grad ./ a.data
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

function zero_grad!(a::Tensor)
    a.grad = zeros(Float32, size(a.grad))
end

function zero_grad!!(a::Tensor)
    nodes = topological_sort(a)
    for node in nodes
        zero_grad!(node)
    end
end

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

function backward(a::Tensor)
    nodes = topological_sort(a)
    a.grad = ones(Float32, size(a.grad))
    for node in reverse(nodes)
        node.require_grad && node.update!()
    end
end

end # module AutoDiff
