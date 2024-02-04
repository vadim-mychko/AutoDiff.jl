export ReLU, Sigmoid, Tanh, Sequential, Linear

abstract type Layer end

# ===================================== ReLU =========================================
struct ReLU <: Layer end
Base.show(io::IO, ::ReLU) = print(io, "ReLU")
(::ReLU)(x::Tensor) = relu(x)

# ==================================== Sigmoid ========================================
struct Sigmoid <: Layer end
Base.show(io::IO, ::Sigmoid) = print(io, "Sigmoid")
(::Sigmoid)(x::Tensor) = sigmoid(x)

# ===================================== Tanh =========================================
struct Tanh <: Layer end
Base.show(io::IO, ::Tanh) = print(io, "Tanh")
(::Tanh)(x::Tensor) = tanh(x)

# =================================== Sequential =====================================
mutable struct Sequential <: Layer
    layers::Vector{Layer}
end

function Sequential(layers::Layer...)
    return Sequential(collect(layers))
end

function Base.show(io::IO, seq::Sequential)
    print(io, "Sequential($(seq.layers))")
end

function (seq::Sequential)(x::Tensor)
    for layer in seq.layers
        x = layer(x)
    end

    return x
end

# ===================================== Linear =======================================
mutable struct Linear <: Layer
    parameters::Tensor
end

function Linear(in::Integer, out::Integer; bias=false)
    bias && (in += 1)
    params = Tensor(rand(Float64, (out, in)); require_grad=true)
    return Linear(params)
end

function Base.show(io::IO, layer::Linear)
    out, in = size(layer.parameters.data)
    print(io, "Linear $out x $in")
end

function (layer::Linear)(x::Tensor)
    return matmul(layer.parameters, x)
end
