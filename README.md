# AudioLoaders

DataLoaders to generate mini-batches in either time-series or scaled spectrogram of audio files. 

## Usage
Let's create some dummy audio files.
```julia
using WAV

temppath = mktempdir()
n = 10
fs = 4800
for i ∈ 1:n
    freq = 100 * rand()
    y = sin.((0:fs-1) / fs * 2π * freq)
    wavwrite(y, joinpath(temppath, "test$(i).wav"), Fs=fs)
end
wavpaths = readdir.(temppath; join=true)
```
Get `AudioLoader` that generates mini-batches in time-series of the audio files.
```julia
using AudioLoaders

m = 3
probs = rand(m, n)
probs ./= sum(probs; dims=1)
data = (wavpaths, probs)
tsconfig = TSConfig(4800, true)
audio_loader = AudioLoader(data,
                           tsconfig; 
                           batchsize=2,
                           partial=false)
for (i, loader) ∈ enumerate(audio_loader)
    xs = first(loader)
    for x ∈ xs
        print("Size of the mini-batch at $(i) iteration: $(size(x))")
        print(" ")
    end
    println("")
end
```

Get `AudioLoader` that generates mini-batches in scaled spectrogram of the audio files.
```julia
specconfig = SpecConfig(512, 256, Windows.hanning(512))
audio_loader = AudioLoader(data,
                           specconfig; 
                           batchsize=2,
                           partial=false)
for (i,loader) ∈ enumerate(audio_loader)
    xs, ls = first(loader)
    for x ∈ xs
         print("Size of the mini-batch at $(i) iteration: $(size(x))")
    end
    timelen = repr.(ls, context=:compact => true) .* " sec"
    print(", time length: $(Tuple(timelen))")
    println("")
end
```

`ndata ` augmented versions of the same data can be generated by providing an augmentation function `augment`.
```julia
augment(x) = rand() > 0.5 ? -x : x
nchannels = 1
ndata = 3
tsconfig = TSConfig(4800, true, augment, nchannels, ndata)
audio_loader = AudioLoader(data,
                           tsconfig; 
                           batchsize=2,
                           partial=false)
for (i,loader) ∈ enumerate(audio_loader)
    xs = first(loader)
    for x ∈ xs
        print("Size of the mini-batch at $(i) iteration: $(size(x))")
        print(" ")
    end
    println("")
end
```  