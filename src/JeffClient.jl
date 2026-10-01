module JeffClient

import ONNXRunTime
import Downloads
import JSON
import Scratch
using LinearAlgebra
import LoopVectorization
@static if Sys.isapple() && Sys.ARCH === :aarch64
    import AppleAccelerate
end

export ChoiceQuestion, NoulQuestion, ScoreQuestion, ONNXBackend, decide
export resolve_checkpoint, load_export
export NativeBackend, logits

include("questions.jl")
include("onnx.jl")
include("hub.jl")
include("safetensors.jl")
include("cpu_settings.jl")
include("native.jl")
include("native_cpu.jl")

__init__() = initialize_cpu!()

end # module JeffClient
