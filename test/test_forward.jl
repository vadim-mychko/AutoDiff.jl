@testset "Addition" begin
    @testset "Scalar" begin
        a = Tensor(2.0)
        b = Tensor(3.0)
        result = a + b
        @test result.data ≈ [5.0]
    end

    @testset "Vector" begin
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a + b
        @test result.data == [5, 7, 9]
    end

    @testset "Matrix" begin
        a = Tensor([1 2; 3 4])
        b = Tensor([5 6; 7 8])
        result = a + b
        @test result.data == [6 8; 10 12]
    end

    @testset "Mixed type" begin
        a = Tensor(2.5)
        b = Tensor(2)
        result = a + b
        @test result.data ≈ [4.5]
    end

    @testset "Broadcast" begin
        a = Tensor([10 5; -5 0])
        b = Tensor([2, 3])
        result = a + b
        @test result.data == [12 7; -2 3]
    end
end
