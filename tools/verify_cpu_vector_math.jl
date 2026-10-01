using JeffClient, LoopVectorization, Test
@assert Base.get_extension(JeffClient, :JeffClientLoopVectorizationExt) !== nothing

@testset "Portable vector activation guards and ownership" begin
    withenv("JEFF_CPU_PORTABLE_VECTOR_MATH" => "1", "JEFF_CPU_VECTOR_MATH_BLOCKS" => "0") do
        for n in (1, 7, 8, 9, 101), width in (1, 128)
            gate = reshape(0.1f0 .+ sin.(Float32.(1:(width*n))), width, n)
            up = reshape(0.2f0 .+ cos.(Float32.(1:(width*n))), width, n)
            expected = JeffClient.native_silu.(gate)
            actual = copy(gate)
            @test JeffClient.cpu_portable_silu!(actual)
            @test actual ≈ expected atol=2e-6 rtol=2e-6
            saved_up = copy(up)
            actual = copy(gate)
            @test JeffClient.cpu_portable_gate!(actual, up)
            @test actual ≈ expected .* up atol=2e-6 rtol=2e-6
            @test up == saved_up
        end
        for x in (
            0.0f0,
            -0.0f0,
            Float32(NaN),
            Float32(Inf),
            -Float32(Inf),
            floatmax(Float32),
            -floatmax(Float32),
            nextfloat(0.0f0),
            -100.0f0,
            100.0f0,
        )
            gate = reshape(Float32[1, x, -1], 3, 1)
            up = fill(2.0f0, 3, 1)
            actual = copy(gate)
            @test !JeffClient.cpu_portable_silu!(actual)
            @test isequal(actual, gate)
            @test isequal(
                JeffClient.native_cpu_silu!(actual),
                JeffClient.native_silu.(gate),
            )
            actual = copy(gate)
            @test !JeffClient.cpu_portable_gate!(actual, up)
            @test isequal(actual, gate)
            @test isequal(
                JeffClient.cpu_owned_mlp_gate!(actual, up),
                JeffClient.native_silu.(gate) .* up,
            )
        end
        for x in
            (0.0f0, -0.0f0, Float32(NaN), Float32(Inf), floatmax(Float32), nextfloat(0.0f0))
            gate = fill(1.0f0, 3, 1)
            up = fill(x, 3, 1)
            @test !JeffClient.cpu_portable_gate!(gate, up)
            @test gate == ones(Float32, 3, 1)
        end
        gate = fill(1.0f0, 3, 1)
        @test !JeffClient.cpu_portable_gate!(gate, gate)
        @test !JeffClient.cpu_portable_gate!(gate, zeros(Float32, 2, 1))
        @test JeffClient.cpu_portable_silu!(zeros(Float32, 0, 1))
        @test JeffClient.cpu_portable_gate!(zeros(Float32, 0, 1), zeros(Float32, 0, 1))
    end
end

import JSON
@testset "Blockwise SiLU retains exceptional scalar semantics" begin
    withenv("JEFF_CPU_PORTABLE_VECTOR_MATH" => "1", "JEFF_CPU_VECTOR_MATH_BLOCKS" => "1") do
        for n in (1, 255, 256, 257, 513, 1024)
            for exception in (
                0.0f0,
                -0.0f0,
                Float32(NaN),
                Float32(Inf),
                -Float32(Inf),
                floatmax(Float32),
                -100.0f0,
                nextfloat(0.0f0),
            )
                input = reshape(0.1f0 .+ sin.(Float32.(1:n)), n, 1)
                input[min(n, 257)] = exception
                expected = JeffClient.native_silu.(input)
                actual = copy(input)
                @test JeffClient.cpu_portable_silu!(actual)
                for i in eachindex(actual)
                    if !isfinite(expected[i]) || iszero(expected[i])
                        @test isequal(actual[i], expected[i])
                    else
                        @test isapprox(actual[i], expected[i]; atol = 2e-6, rtol = 2e-6)
                    end
                end
            end
        end
    end
end
withenv("JEFF_CPU_PORTABLE_VECTOR_MATH" => "1") do
    include(joinpath(@__DIR__, "..", "test", "native.jl"))
end
