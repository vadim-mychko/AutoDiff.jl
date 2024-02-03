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

    @testset "Unary `$op` on random tensors" for op in [-, inv, sin, cos, exp, log]
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

@testset "Backward pass" begin
    @testset "Simple backpropagation, n_dims = $(n_dims)" for n_dims in DIM_RANGE
        require_grad = true
        dims = Tuple(rand(N_ELEMENTS) for _ in 1:n_dims)
        a = Tensor(rand(Float32, dims); require_grad)
        b = Tensor(rand(Float32, dims); require_grad)
        c = a + b
        d = Tensor(rand(Float32, dims); require_grad)
        e = c * d
        backward(e)
        @test e.grad ≈ ones(Float32, dims)
        @test d.grad ≈ c.data
        @test c.grad ≈ d.data
        @test a.grad ≈ c.grad
        @test b.grad ≈ c.grad
    end

    @testset "Complex backpropagation, n_dims = $(n_dims)" for n_dims in DIM_RANGE
        require_grad = true
        dims = Tuple(rand(N_ELEMENTS) for _ in 1:n_dims)
        a = Tensor(rand(Float32, dims); require_grad)
        b = Tensor(rand(Float32, dims); require_grad)
        c = a + b
        d = a * c
        backward(d)
        @test d.grad ≈ ones(Float32, dims)
        @test c.grad ≈ a.data
        @test b.grad ≈ c.grad
        @test a.grad ≈ c.grad .+ c.data
    end

    @testset "Complex backpropagation with `matmul`" begin
        require_grad = true
        dims = Tuple(rand(N_ELEMENTS) for _ in 1:3)
        a = Tensor(rand(Float32, dims[1:2]); require_grad)
        b = Tensor(rand(Float32, dims[1:2]); require_grad)
        c = a * b
        d = Tensor(rand(Float32, dims[2:3]); require_grad)
        e = matmul(c, d)
        backward(e)
        @test e.grad ≈ ones(Float32, dims[[1, 3]])
        @test c.grad ≈ e.grad * transpose(d.data)
        @test d.grad ≈ transpose(c.data) * e.grad
        @test b.grad ≈ c.grad .* a.data
        @test a.grad ≈ c.grad .* b.data
    end
end
