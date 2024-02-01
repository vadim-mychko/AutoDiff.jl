module AutoDiff

mutable struct Tensor{T <: Real, N <: Integer}
    data::Array{T, N}
    grad::Array{T, N}
    label::String
    operation::String
    parents::Set{Tensor}
    require_grad::Bool
end

end # module AutoDiff
