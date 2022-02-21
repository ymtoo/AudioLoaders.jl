module AudioLoaders

using Reexport

using Distributions
using DSP
@reexport using DSP: Windows, stft, power, pow2db
using FLoops
using Flux: Data._nobs, Data._getobs, cpu, gpu
using ImageTransformations
using ProgressMeter
using SignalAnalysis
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
using WAV

include("mel.jl")
include("config.jl")
include("audioloader.jl")
include("utils.jl")
include("embeddings.jl")
include("augmentor.jl")

const sampletype = Float32

end # module
