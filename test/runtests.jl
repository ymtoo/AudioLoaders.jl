using AudioLoaders

using DelimitedFiles, Distributions, Test, WAV

paths = readdir("data/audio/"; join=true, sort=true)
metadata = readdlm("data/metadata.csv", ','; skipstart=1)
files = metadata[:,1]
indices = sortperm(files)
paths = paths[indices]
labeltype = Int
labels = convert.(labeltype, metadata[indices,2])
probtype = Float32
probs = convert.(probtype, metadata[indices,3])
data = (paths, labels, probs)
wavlens, samplingrates = begin
    tls = AudioLoaders.sampletype[]
    srs = AudioLoaders.sampletype[]
    for path ∈ paths
        n = first(wavread(path; format="size"))
        fs = (wavread(path; subrange=1:1))[2]
        push!(tls, n / fs)
        push!(srs, fs)
    end
    tls, srs
end

@testset "data = (paths, labels, probs)
AudioLoader" begin
    ncs = [1]
    for nc ∈ ncs
        tsconfig = TSConfig(winsize = 4800, 
                            randsegment = false, 
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
                                            partial=partial,
                                            shuffle=false)
                    audio_loader1 = audio_loader[2:end] # warning
                    n = length(paths) / batchsize
                    @test length(audio_loader) == (partial ? ceil(Int, n) : floor(Int, n))
                    for (i, X) ∈ enumerate(audio_loader)
                        i > 1 && !isempty(audio_loader1) && (@test X == audio_loader1[i-1]) # getindex
                        
                        # test targets
                        startindex = (i-1) * audio_loader.batchsize + 1
                        @test eltype(X[2]) == labeltype
                        @test eltype(X[3]) == probtype
                        if !partial | (i < length(audio_loader))
                            stopindex = startindex + audio_loader.batchsize - 1
                            @test X[2] == labels[startindex:stopindex]
                            @test X[3] == probs[startindex:stopindex]
                        else
                            @test X[2] == labels[startindex:end]
                            @test X[3] == probs[startindex:end]
                        end

                        # test data
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
                            X1s, tls, srs = first(X)
                            if !partial | (i < length(audio_loader))
                                for X1 ∈ X1s 
                                    @test size(X1) == (config.newdims..., nc, batchsize)
                                end
                                stopindex = startindex + batchsize - 1
                                @test wavlens[startindex:stopindex] ≈ tls 
                                @test samplingrates[startindex:stopindex] ≈ srs
                            else
                                @test wavlens[startindex:end] ≈ tls
                                @test samplingrates[startindex:end] ≈ srs
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

@testset "embeddings" begin
    
    tsconfig = TSConfig(winsize = 4800, 
                        randsegment = false, 
                        augment = identity,
                        nchannels = 1,
                        ndata = 1)
    specconfig = SpecConfig(winsize = 512,
                            noverlap = 256,
                            window = Windows.hanning(512),
                            scaled = x -> pow2db.(abs2.(x)),
                            augment = identity,
                            newdims = (100,100),
                            nchannels = 1,
                            ndata = 1)
    tsloader = AudioLoader(data,
                           tsconfig; 
                           batchsize=1,
                           shuffle=false,
                           partial=true)
    specloader = AudioLoader(data,
                             specconfig; 
                             batchsize=1,
                             shuffle=false,
                             partial=true)
    function f1(x::AbstractArray{T}) where {T} 
        dropdims(sum(x; dims=[1,2]); dims=(1,2))
    end
    function f2(x::AbstractArray{T}, y) where {T} 
        z1 = dropdims(sum(x; dims=[1,2]); dims=(1,2))
        cat(z1, y[1:1,:]; dims=1)
    end
    Z1 = embed(f1, tsloader)
    @test size(Z1) == (1, length(paths))
    Z2 = embed(f2, specloader)
    @test size(Z2) == (2, length(paths))
end

@testset "augmentor" begin
    
    n = 9600
    x = randn(n)
    @test apply(Amplify(Uniform(1.999999,2.000001)), x) ≈ 2 .* x atol=1e-3
    @test apply(PolarityInverse(), x) == -x
    @test apply(CircularShift(Binomial(1,1)), x) == circshift(x, 1)
    @test apply(TimeStretch(0.0000001), x) ≈ x atol=0.1
    @test apply(PitchShift(0.0000001), x) ≈ x atol=0.1

    @test random_apply(PolarityInverse(), x; p=0) == x
    @test random_apply(PolarityInverse(), x; p=1) == -x

end