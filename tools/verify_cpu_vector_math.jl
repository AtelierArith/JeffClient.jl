using JeffClient, LoopVectorization, Test
using QwenDecisionCore
@assert Base.get_extension(JeffClient, :JeffClientLoopVectorizationExt) !== nothing

@testset "Portable vector activation guards and ownership" begin
    QwenDecisionCore.with_cpu_settings(
        :portable_vector_math => "1",
        :vector_math_blocks => "0",
    ) do
        for n in (1, 7, 8, 9, 101), width in (1, 128)
            gate = reshape(0.1f0 .+ sin.(Float32.(1:(width*n))), width, n)
            up = reshape(0.2f0 .+ cos.(Float32.(1:(width*n))), width, n)
            expected = QwenDecisionCore.native_silu.(gate)
            actual = copy(gate)
            @test QwenDecisionCore.cpu_portable_silu!(actual)
            @test actual ≈ expected atol=2e-6 rtol=2e-6
            saved_up = copy(up)
            actual = copy(gate)
            @test QwenDecisionCore.cpu_portable_gate!(actual, up)
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
            @test !QwenDecisionCore.cpu_portable_silu!(actual)
            @test isequal(actual, gate)
            @test isequal(
                QwenDecisionCore.native_cpu_silu!(actual),
                QwenDecisionCore.native_silu.(gate),
            )
            actual = copy(gate)
            @test !QwenDecisionCore.cpu_portable_gate!(actual, up)
            @test isequal(actual, gate)
            @test isequal(
                QwenDecisionCore.cpu_owned_mlp_gate!(actual, up),
                QwenDecisionCore.native_silu.(gate) .* up,
            )
        end
        for x in
            (0.0f0, -0.0f0, Float32(NaN), Float32(Inf), floatmax(Float32), nextfloat(0.0f0))
            gate = fill(1.0f0, 3, 1)
            up = fill(x, 3, 1)
            @test !QwenDecisionCore.cpu_portable_gate!(gate, up)
            @test gate == ones(Float32, 3, 1)
        end
        gate = fill(1.0f0, 3, 1)
        @test !QwenDecisionCore.cpu_portable_gate!(gate, gate)
        @test !QwenDecisionCore.cpu_portable_gate!(gate, zeros(Float32, 2, 1))
        @test QwenDecisionCore.cpu_portable_silu!(zeros(Float32, 0, 1))
        @test QwenDecisionCore.cpu_portable_gate!(
            zeros(Float32, 0, 1),
            zeros(Float32, 0, 1),
        )
    end
end

import JSON
@testset "Blockwise SiLU retains exceptional scalar semantics" begin
    QwenDecisionCore.with_cpu_settings(
        :portable_vector_math => "1",
        :vector_math_blocks => "1",
    ) do
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
                expected = QwenDecisionCore.native_silu.(input)
                actual = copy(input)
                @test QwenDecisionCore.cpu_portable_silu!(actual)
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
@testset "Blockwise MLP gates preserve exceptional values and ownership" begin
    QwenDecisionCore.with_cpu_settings(
        :portable_vector_math => true,
        :vector_math_blocks => true,
    ) do
        for n in (255, 256, 257, 513),
            exception in
            (0.0f0, -0.0f0, Float32(NaN), Float32(Inf), -100.0f0, nextfloat(0.0f0))

            gate = reshape(0.1f0 .+ sin.(Float32.(1:n)), n, 1)
            up = reshape(0.2f0 .+ cos.(Float32.(1:n)), n, 1)
            gate[min(n, 257)] = exception
            expected = QwenDecisionCore.native_silu.(gate) .* up
            saved_up = copy(up)
            @test QwenDecisionCore.cpu_portable_gate!(gate, up)
            @test up == saved_up
            for i in eachindex(gate)
                @test !isfinite(expected[i]) || iszero(expected[i]) ?
                      isequal(gate[i], expected[i]) :
                      isapprox(gate[i], expected[i]; atol = 2e-6, rtol = 2e-6)
            end
        end
        gate = reshape(repeat(Float32[0, -0], 256), 512, 1)
        up = reshape(repeat(Float32[1, -1, Inf, NaN], 128), 512, 1)
        expected = QwenDecisionCore.native_silu.(gate) .* up
        @test QwenDecisionCore.cpu_portable_gate!(gate, up)
        @test isequal(gate, expected)
        @test !QwenDecisionCore.cpu_portable_gate!(gate, gate)
    end
end
QwenDecisionCore.with_cpu_settings(:portable_vector_math => "1") do
    include(joinpath(@__DIR__, "..", "test", "native.jl"))
end

@testset "Portable softmax masks, subnormals and nonfinite fallback" begin
    QwenDecisionCore.with_cpu_settings(:portable_vector_math => true) do
        for n in (1, 7, 64, 257)
            scores = reshape(sin.(Float32.(1:(3n))), n, 3)
            scores[:, 2] .= -floatmax(Float32)
            n > 1 && (scores[2:end, 3] .= -90.0f0)
            expected = exp.(scores .- maximum(scores; dims = 1))
            expected ./= sum(expected; dims = 1)
            actual = copy(scores)
            @test QwenDecisionCore.cpu_portable_softmax!(actual)
            @test actual ≈ expected atol = 2e-6 rtol = 2e-6
            @test all(actual[2:end, 3] .> 0.0f0)
            @test actual[:, 2] == fill(1.0f0 / n, n)
        end
        for x in (NaN32, Inf32, -Inf32)
            scores = Float32[1 x; 2 3]
            saved = copy(scores)
            @test !QwenDecisionCore.cpu_portable_softmax!(scores)
            @test isequal(scores, saved)
        end
    end
end

@testset "Parallel activation block boundaries and exceptional lanes" begin
    QwenDecisionCore.with_cpu_settings(
        :portable_vector_math => true,
        :vector_math_blocks => true,
    ) do
        for n in (65535, 65536, 65537, 1048577)
            gate = reshape(sin.(Float32.(1:n)), n, 1)
            gate[1:(n÷2)] .= -0.0f0
            gate[[257, n-1, n]] .= Float32[NaN, nextfloat(0.0f0), -100]
            expected = QwenDecisionCore.native_silu.(gate)
            actual = copy(gate)
            @test QwenDecisionCore.cpu_portable_silu!(actual)
            matches(x, y) =
                !isfinite(y) || iszero(y) ? isequal(x, y) :
                isapprox(x, y; atol = 2e-6, rtol = 2e-6)
            @test all(matches(x, y) for (x, y) in zip(actual, expected))
            up = reshape(cos.(Float32.(1:n)), n, 1)
            up[[256, n-2]] .= Float32[Inf, NaN]
            saved_up = copy(up)
            expected .*= up
            @test QwenDecisionCore.cpu_portable_gate!(gate, up)
            @test all(matches(x, y) for (x, y) in zip(gate, expected))
            @test isequal(up, saved_up)
        end
    end
end
