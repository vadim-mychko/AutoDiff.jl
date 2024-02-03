using AutoDiff
using Test

const N_ELEMENTS = 1:10
const DIM_RANGE = 1:3

@testset "Forward pass" begin
    @testset "Binary `$op` on random tensors" for op in [+, -, *, /]
        for n_dims in DIM_RANGE
            dims = Tuple(rand(N_ELEMENTS) for _ in 1:n_dims)
            a = rand(Float32, dims)
            b = rand(Float32, dims)
            ta = Tensor(a)
            tb = Tensor(b)
            @test op(ta, tb).data ≈ op.(a, b)
        end
    end

    @testset "Binary `matmul` on random tensors" begin
        dims = Tuple(rand(N_ELEMENTS) for _ in 1:3)
        a = rand(Float32, dims[1:2])
        b = rand(Float32, dims[2:3])
        ta = Tensor(a)
        tb = Tensor(b)
        @test matmul(ta, tb).data ≈ a * b
    end

    @testset "Unary `$op` on random tensors" for op in [-, inv]
        for n_dims in DIM_RANGE
            dims = Tuple(rand(N_ELEMENTS) for _ in 1:n_dims)
            a = rand(Float32, dims)
            ta = Tensor(a)
            @test op(ta).data ≈ op.(a)
        end
    end

    @testset "Unary `^` on random tensors" begin
        for n_dims in DIM_RANGE
            dims = Tuple(rand(N_ELEMENTS) for _ in 1:n_dims)
            a = rand(Float32, dims)
            b = rand()
            ta = Tensor(a)
            @test (ta ^ b).data ≈ (a .^ b)
        end
    end
end
