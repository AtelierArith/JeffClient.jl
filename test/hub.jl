@testset "Julia checkpoint cache" begin
    mktempdir() do root
        commit = repeat("a", 40)
        repo = "test/jeff-cache"
        endpoint = joinpath(root, "endpoint")
        cache = joinpath(root, "cache")
        shared = joinpath(root, "shared")
        files = [
            "config.json",
            "decision_config.json",
            "readout.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ]
        api = joinpath(endpoint, "api", "models", repo, "revision", "main")
        mkpath(dirname(api))
        write(
            api,
            JSON.json(
                Dict(
                    "sha" => commit,
                    "siblings" =>
                        [Dict("rfilename" => name) for name in [files; "videos/demo.mp4"]],
                ),
            ),
        )
        source = joinpath(endpoint, repo, "resolve", commit)
        mkpath(source)
        for file in files
            write(joinpath(source, file), "fixture")
        end
        withenv(
            "HF_ENDPOINT" => "file://$endpoint",
            "HF_HUB_CACHE" => shared,
            "HF_HUB_OFFLINE" => "0",
            "HF_TOKEN" => nothing,
        ) do
            dir = resolve_checkpoint(repo; cache_dir = cache)
            @test dir == joinpath(cache, "models--test--jeff-cache", "snapshots", commit)
            @test Set(readdir(dir)) == Set(files)
            @test read(
                joinpath(cache, "models--test--jeff-cache", "refs", "main"),
                String,
            ) == commit
            @test resolve_checkpoint(dir) == dir
            @test resolve_checkpoint(repo; cache_dir = cache, offline = true) == dir
            @test resolve_checkpoint(
                repo;
                revision = commit,
                cache_dir = cache,
                offline = true,
            ) == dir
            @test_throws ArgumentError resolve_checkpoint(
                "test/missing";
                cache_dir = cache,
                offline = true,
            )
            @test_throws ArgumentError resolve_checkpoint(
                repo;
                revision = "../outside",
                cache_dir = cache,
            )
            @test_throws ArgumentError resolve_checkpoint("./missing")
            @test_throws ArgumentError resolve_checkpoint(mktempdir(root))

            # Existing HF snapshots take precedence over the package cache.
            shared_dir = joinpath(shared, "models--test--jeff-cache", "snapshots", commit)
            mkpath(dirname(shared_dir))
            cp(dir, shared_dir)
            @test resolve_checkpoint(
                repo;
                revision = commit,
                cache_dir = cache,
                offline = true,
            ) == shared_dir

            # Exercise default Scratch storage, using a unique fixture repo.
            scratch_repo = "fixture/jeff-" * basename(root)
            scratch_api =
                joinpath(endpoint, "api", "models", scratch_repo, "revision", "main")
            mkpath(dirname(scratch_api))
            cp(api, scratch_api)
            scratch_source = joinpath(endpoint, scratch_repo, "resolve", commit)
            mkpath(dirname(scratch_source))
            cp(source, scratch_source)
            scratch_dir = resolve_checkpoint(scratch_repo)
            @test startswith(scratch_dir, joinpath(first(DEPOT_PATH), "scratchspaces"))
            @test resolve_checkpoint(scratch_repo; offline = true) == scratch_dir
        end
    end
end

@testset "Interrupted download and resume" begin
    mktempdir() do root
        commit = repeat("b", 40)
        repo = "test/resume"
        endpoint = joinpath(root, "endpoint")
        cache = joinpath(root, "cache")
        files = [
            "config.json",
            "decision_config.json",
            "readout.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ]
        api = joinpath(endpoint, "api", "models", repo, "revision", "main")
        mkpath(dirname(api))
        write(
            api,
            JSON.json(
                Dict(
                    "sha" => commit,
                    "siblings" => [Dict("rfilename" => name) for name in files],
                ),
            ),
        )
        source = joinpath(endpoint, repo, "resolve", commit)
        mkpath(source)
        for file in files[1:(end-1)]
            write(joinpath(source, file), "fixture")
        end
        withenv(
            "HF_ENDPOINT" => "file://$endpoint",
            "HF_HUB_CACHE" => joinpath(root, "shared"),
            "HF_HUB_OFFLINE" => "0",
            "HF_TOKEN" => nothing,
        ) do
            @test_throws Exception resolve_checkpoint(repo; cache_dir = cache)
            cache_root = joinpath(cache, "models--test--resume")
            @test !isfile(joinpath(cache_root, "refs", "main"))
            @test all(!startswith(name, "download-") for name in readdir(cache_root))
            @test_throws ArgumentError resolve_checkpoint(
                repo;
                cache_dir = cache,
                offline = true,
            )
            # Resuming preserves files already fully downloaded.
            completed = joinpath(cache_root, "snapshots", commit, "config.json")
            write(completed, "keep this cached file")
            write(joinpath(source, "model.safetensors"), "fixture")
            dir = resolve_checkpoint(repo; cache_dir = cache)
            @test read(joinpath(dir, "config.json"), String) == "keep this cached file"
            @test resolve_checkpoint(repo; cache_dir = cache, offline = true) == dir
        end
    end
end

@testset "Export metadata calibration" begin
    mktempdir() do dir
        cp(MODEL, joinpath(dir, "model.onnx"))
        write(
            joinpath(dir, "export_config.json"),
            JSON.json(
                Dict(
                    "format_version" => 1,
                    "model_file" => "model.onnx",
                    "output_name" => "logits",
                ),
            ),
        )
        write(
            joinpath(dir, "decision_config.json"),
            JSON.json(
                Dict("format_version" => 1, "temperature" => 2.0, "max_options" => 2),
            ),
        )
        backend = load_export(dir)
        try
            result = decide(backend, Dict("scores" => Float32[0 log(9)]), NoulQuestion())
            @test result.noul ≈ 0.75 rtol=1e-6
            @test backend.max_options == 2
        finally
            close(backend)
        end
        write(
            joinpath(dir, "export_config.json"),
            JSON.json(
                Dict(
                    "format_version" => 1,
                    "model_file" => "../model.onnx",
                    "output_name" => "logits",
                ),
            ),
        )
        @test_throws ArgumentError load_export(dir)
    end
end
