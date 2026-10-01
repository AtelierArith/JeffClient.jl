using JeffClient, Octavian, LinearAlgebra, Test, InteractiveUtils
@assert Base.get_extension(JeffClient, :JeffClientOctavianExt) !== nothing

@testset "Worker-local serial Octavian state products" begin
    for width in (8, 128), n in (1, 37, 64, 128), beta in (0.0f0, 1.0f0)
        state = reshape(sin.(Float32.(1:(width*width))), width, width)
        storage = reshape(cos.(Float32.(1:(width*2*n))), width, 2, n)
        view_rhs = @view storage[:, 2, :]
        for rhs in (view_rhs, copy(view_rhs), transpose(permutedims(view_rhs)))
            output = fill(beta == 0 ? NaN32 : 0.5f0, width, n)
            expected = copy(output)
            mul!(expected, state, rhs, -1.0f0, beta)
            original_state, original_rhs = copy(state), copy(rhs)
            JeffClient.with_cpu_settings(:octavian_delta => "1") do
                JeffClient.cpu_delta_state_product!(output, state, rhs, -1.0f0, beta)
            end
            @test output ≈ expected atol=2e-5 rtol=2e-5
            @test state == original_state
            @test rhs == original_rhs
        end
    end
end

JeffClient.with_cpu_settings(:octavian_delta => "1") do
    a = ones(Float32, 128, 128)
    b = ones(Float32, 128, 64)
    c = similar(b)
    @code_warntype JeffClient.cpu_delta_state_product!(c, a, b, 1.0f0, 0.0f0)
end
