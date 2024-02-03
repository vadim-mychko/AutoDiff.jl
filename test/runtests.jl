using AutoDiff
using Test

@testset "Forward pass" begin
    include("test_forward.jl")
end

@testset "Backward pass" begin
    include("test_backward.jl")
end
