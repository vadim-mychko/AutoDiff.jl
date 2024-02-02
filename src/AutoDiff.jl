module AutoDiff

import Base

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
    require_grad::Bool = false
)
    return Tensor(data, label, operation, parents, require_grad,
                  zeros(Float32, size(data)), () -> return)
end

function Base.show(io::IO, obj::Tensor)
    print(io, "Tensor(")
    show(io, obj.data)
    !isempty(obj.label) && print(io, ", label=\"$(obj.label)\"")
    !isempty(obj.operation) && print(io, ", operation=\"$(obj.operation)\"")
    obj.require_grad && print(io, ", require_grad=true")
    print(io, ")")
end

end # module AutoDiff
