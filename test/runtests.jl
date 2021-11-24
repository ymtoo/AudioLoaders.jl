using AudioLoaders

using DelimitedFiles, Test, WAV

paths = readdir("data/audio/"; join=true, sort=true)
metadata = readdlm("data/metadata.csv", ','; skipstart=1)
files = metadata[:,1]
indices = sortperm(files)
paths = paths[indices]
labels = convert.(Int, metadata[indices,2])
probs = convert.(Float32, metadata[indices,3])
data = (paths, labels, probs)
wavlens = begin
    ls = AudioLoaders.sampletype[]
    for path ∈ paths
        n = first(wavread(path; format="size"))
        fs = (wavread(path; subrange=1:1))[2]
        push!(ls, n / fs)
    end
    ls
end

@testset "data = (paths, labels, probs)
AudioLoader" begin
    ncs = [1]
    for nc ∈ ncs
        tsconfig = TSConfig(winsize = 4800, 
                            randsegment = true, 
                            augment = identity,
                            nchannels = nc,
                            ndata = 1)
        specconfig = SpecConfig(winsize = 512,
                                noverlap = 256,
                                window = Windows.hanning(512),
                                scaled = x -> pow2db.(abs2.(x)),
                                augment = identity,
                                newdims = (100,100),
                                nchannels = nc,
                                ndata = 1)
        configs = [tsconfig,specconfig]
        batchsizes = [1,2]
        partials = [true,false]
        for config ∈ configs
            for batchsize ∈ batchsizes
                for partial ∈ partials
                    audio_loader = AudioLoader(data,
                                            config; 
                                            batchsize=batchsize,
                                            partial=partial)
                    n = length(paths) / batchsize
                    @test length(audio_loader) == (partial ? ceil(Int, n) : floor(Int, n))
                    for (i, X) ∈ enumerate(audio_loader)
                        startindex = (i-1) * batchsize + 1
                        if config isa TSConfig
                            for X1 ∈ first(X)
                                if !partial | (i < length(audio_loader))
                                    @test size(X1) == (config.winsize, 1, nc, batchsize)
                                else
                                    @test size(X1) == (config.winsize, 
                                                       1, 
                                                       nc, 
                                                       length(first(audio_loader.data))-startindex+1)
                                end
                            end
                        elseif config isa SpecConfig
                            X1s, ls = first(X)
                            if !partial | (i < length(audio_loader))
                                for X1 ∈ X1s 
                                    @test size(X1) == (config.newdims..., nc, batchsize)
                                end
                                stopindex = startindex + batchsize - 1
                                @test wavlens[startindex:stopindex] ≈ ls
                            else
                                @test wavlens[startindex:end] ≈ ls
                                for X1 ∈ X1s 
                                    @test size(X1) == (config.newdims..., nc, length(first(audio_loader.data))-startindex+1)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end