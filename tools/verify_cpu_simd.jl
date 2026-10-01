using SIMD, JeffClient, Test
@assert Base.get_extension(JeffClient, :JeffClientSIMDExt) !== nothing

@testset "SIMD convolution versus ordered scalar reference" begin
    for channels in (1, 7, 8, 9, 128, 6144), tokens in (0, 1, 9), taps in (1, 4)
        input = reshape(sin.(Float32.(1:(channels*tokens))), channels, tokens)
        coefficients = reshape(cos.(Float32.(1:(channels*taps))), channels, taps)
        weight = transpose(coefficients)
        expected = zeros(Float32, size(input))
        for tap = 1:taps, token = max(1, taps-tap+1):tokens, channel = 1:channels
            expected[channel, token] +=
                input[channel, token-(taps-tap)] * coefficients[channel, tap]
        end
        original = copy(input)
        actual = similar(expected)
        withenv("JEFF_CPU_SIMD" => "1") do
            JeffClient.cpu_convolution!(actual, input, weight)
        end
        @test actual == expected
        @test input == original
    end
end
