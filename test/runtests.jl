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

@testset "AudioLoader" begin
    ncs = [1]
    for nc ∈ ncs
        tsconfig = TSConfig(winsize = 4800, 
                            randsegment = false, 
                            preprocess_augment = (x,fs) -> identity(x),
                            nchannels = nc,
                            ndata = 1)
        specconfig = SpecConfig(winsize = 512,
                                noverlap = 256,
                                window = Windows.hanning(512),
                                scaled = x -> pow2db.(abs2.(x)),
                                preprocess_augment = (x,fs) -> identity(x),
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
                        X1s, tls, srs = first(X)
                        if config isa TSConfig
                            for X1 ∈ X1s
                                if !partial | (i < length(audio_loader))
                                    @test size(X1) == (config.winsize, 1, nc, batchsize)
                                    stopindex = startindex + batchsize - 1
                                else
                                    @test size(X1) == (config.winsize, 
                                                    1, 
                                                    nc, 
                                                    length(first(audio_loader.data))-startindex+1)
                                    stopindex = length(wavlens)
                                end
                            end
                        elseif config isa SpecConfig
                            if !partial | (i < length(audio_loader))
                                for X1 ∈ X1s 
                                    @test size(X1) == (config.newdims..., nc, batchsize)
                                end
                                stopindex = startindex + batchsize - 1
                                # @test wavlens[startindex:stopindex] ≈ tls
                                # @test samplingrates[startindex:stopindex] ≈ srs
                            else
                                stopindex = length(wavlens)
                                # @test wavlens[startindex:end] ≈ tls
                                # @test samplingrates[startindex:end] ≈ srs
                                for X1 ∈ X1s 
                                    @test size(X1) == (config.newdims..., nc, length(first(audio_loader.data))-startindex+1)
                                end
                            end
                        end
                        @test wavlens[startindex:stopindex] ≈ tls
                        @test samplingrates[startindex:stopindex] ≈ srs
                    end
                end
            end
        end
    end
end

@testset "utils" begin
    tsconfig = TSConfig(winsize = 4800, 
                            randsegment = false, 
                            preprocess_augment = (x,fs) -> identity(x),
                            nchannels = 1,
                            ndata = 1)
    specconfig = SpecConfig(winsize = 512,
                            noverlap = 256,
                            window = Windows.hanning(512),
                            scaled = x -> pow2db.(abs2.(x)),
                            preprocess_augment = (x,fs) -> identity(x),
                            newdims = (100,100),
                            nchannels = 1,
                            ndata = 1)
    batchsize = 2
    shuffles = [true,false]
    for shuffle ∈ shuffles
        tsloader = AudioLoader(data,
                            tsconfig; 
                            batchsize=batchsize,
                            partial=true,
                            shuffle=shuffle)
        specloader = AudioLoader(data,
                                specconfig; 
                                batchsize=batchsize,
                                partial=true,
                                shuffle=shuffle)
        tstargets = gettargets(tsloader)
        spectargets = gettargets(specloader)
        for (tstarget, spectarget, target) ∈ zip(tstargets, spectargets, (labels, probs))
            @test tstarget == target[tsloader.indices]
            @test spectarget == target[specloader.indices]
        end
    end
end

@testset "mel" begin

    @test hz2mel(60) ≈ 0.9
    @test hz2mel.([110,220,440]) ≈ [1.65,3.3,6.6]
    @test mel2hz(3) ≈ 200
    @test mel2hz.([1,2,3,4,5]) ≈ [66.667,133.333,200.,266.667,333.333] atol=1e-3

    @test fft_frequencies(22050, 16) == [0.,1378.125,2756.25,4134.375,
                                         5512.5,6890.625,8268.75,9646.875,11025.]
    @test mel_frequencies(40, 0, 11025) ≈ [      0.,     85.317,    170.635,    255.952,
                                            341.269,    426.586,    511.904,    597.221,
                                            682.538,    767.855,    853.173,    938.49 ,
                                           1024.856,   1119.114,   1222.042,   1334.436,
                                           1457.167,   1591.187,   1737.532,   1897.337,
                                           2071.84 ,   2262.393,   2470.47 ,   2697.686,
                                           2945.799,   3216.731,   3512.582,   3835.643,
                                           4188.417,   4573.636,   4994.285,   5453.621,
                                           5955.205,   6502.92 ,   7101.009,   7754.107,
                                           8467.272,   9246.028,  10096.408,  11025.   ] atol=1e-2
    
    @test getmelfilters(4800, 8, 4) ≈ [0. 0.00102068 0.         0.         0.;
                                       0. 0.00166109 0.         0.         0.;
                                       0. 0.         0.00187853 0.         0.;
                                       0. 0.         0.00024229 0.00123142 0.] atol=1e-3
end

@testset "embeddings" begin
    
    tsconfig = TSConfig(winsize = 4800, 
                        randsegment = false, 
                        preprocess_augment = (x,fs) -> identity(x),
                        nchannels = 1,
                        ndata = 1)
    specconfig = SpecConfig(winsize = 512,
                            noverlap = 256,
                            window = Windows.hanning(512),
                            scaled = x -> pow2db.(abs2.(x)),
                            preprocess_augment = (x,fs) -> identity(x),
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
    
    function f(x::AbstractArray{T}) where {T} 
        dropdims(sum(x; dims=[1,2]); dims=(1,2))
    end
    function f(x::AbstractArray{T}, y) where {T} 
        z1 = dropdims(sum(x; dims=[1,2]); dims=(1,2))
        cat(z1, y[1:1,:]; dims=1)
    end
    function f(x::AbstractArray{T}, y, z) where {T} 
        z1 = dropdims(sum(x; dims=[1,2]); dims=(1,2))
        cat(z1, y[1:1,:], z[1:1,:]; dims=1)
    end
    for withtl ∈ [(true, log),(false, log)], withsr ∈ [(true, log),(false, log)]
        Z1 = embed(f, tsloader; withtl=withtl, withsr=withsr)
        Z2 = embed(f, specloader; withtl=withtl, withsr=withsr)
        nd1, tlsr = if first(withtl) && first(withsr)
            3, cat(reshape(log.(wavlens), 1, :), reshape(log.(samplingrates), 1, :); dims=1)
        elseif first(withtl) 
            2, reshape(log.(wavlens), 1, :)
        elseif first(withsr)
            2, reshape(log.(samplingrates), 1, :)
        else
            1, nothing
        end
        @test size(Z1) == (nd1, length(paths))
        tlsr !== nothing && (@test Z1[2:end,:] == tlsr)
        @test size(Z2) == (nd1, length(paths))
        tlsr !== nothing && (@test Z2[2:end,:] == tlsr)
    end
end

@testset "augmentor" begin
    
    n = 96000
    x = randn(n)
    @test apply(Amplify(Uniform(1.999999,2.000001)), x) ≈ 2 .* x atol=1e-3
    @test apply(PolarityInverse(), x) == -x
    @test apply(CircularShift(Binomial(1,1)), x) == circshift(x, 1)
    @test apply(TimeStretch(0.00000001), x) ≈ x atol=0.1
    @test apply(PitchShift(0.00000001), x) ≈ x atol=0.1
    @test std(apply(BackgroundNoise(), x) - x) ≈ √2 atol=0.1
    @test std(apply(BackgroundNoise(0), x) - x) ≈ √2 atol=0.1

    @test random_apply(PolarityInverse(), x; p=0) == x
    @test random_apply(PolarityInverse(), x; p=1) == -x

end
