module AutoDiff

import Base: show, +

export Tensor

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
    label::AbstractString = "",
    operation::AbstractString = "",
    parents::AbstractSet{Tensor} = Set{Tensor}(),
    require_grad::Bool = false,
)
    return Tensor(data, label, operation, parents, require_grad,
                  zeros(Float32, size(data)), () -> return)
end

function Tensor(data::Real; kwargs...)
    return Tensor([data]; kwargs...)
end

function show(io::IO, obj::Tensor)
    print(io, "Tensor(")
    show(io, obj.data)
    !isempty(obj.label) && print(io, ", label=\"$(obj.label)\"")
    !isempty(obj.operation) && print(io, ", operation=\"$(obj.operation)\"")
    obj.require_grad && print(io, ", require_grad=true")
    print(io, ")")
end

function reshape_grad(grad::AbstractArray, target_shape::Tuple)
    grad_shape = size(grad)
    grad_shape == target_shape && return grad
    target_shape == () && return sum(grad)

    while length(target_shape) != length(grad_shape)
        push!(target_shape, 1)
    end

    dims = [i for i in 1:length(target_shape)
            if target_shape[i] == 1 && grad_shape[i] != 1]

    return reshape(sum(grad; dims), target_shape)
end

function +(a::Tensor, b::Tensor)
    parents = Set{Tensor}([a, b])
    require_grad = a.require_grad || b.require_grad
    out = Tensor(a.data .+ b.data; operation="+", parents, require_grad)
    out.update! = () -> begin
        a.grad += reshape_grad(out.grad, size(a.grad))
        b.grad += reshape_grad(out.grad, size(b.grad))
    end

    return out
end

+(a::Tensor, b) = a + Tensor(b)
+(a, b::Tensor) = Tensor(a) + b

end # module AutoDiff
