module AudioLoaders

using Reexport

using Distributions
@reexport using DSP: Windows, stft, power, pow2db
using Flux: Data._nobs, Data._getobs, cpu, gpu
using ImageTransformations
using ProgressMeter
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
using WAV

include("config.jl")
include("audioloader.jl")
include("utils.jl")
include("embeddings.jl")
include("augmentor.jl")

const sampletype = Float32

end # module
