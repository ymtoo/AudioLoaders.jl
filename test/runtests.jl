using AudioLoaders

using DelimitedFiles, Distributions, SignalAnalysis, Test, WAV

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
                            preprocess_augment = x -> identity(x),
                            nchannels = nc,
                            ndata = 1)
        specconfig = SpecConfig(winsize = 1024,
                                noverlap = 512,
                                window = Windows.hanning(1024),
                                scaled = (a, fs) -> melscale(a, 1024; fs=fs),
                                preprocess_augment = x -> identity(x),
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
                            preprocess_augment = x -> identity(x),
                            nchannels = 1,
                            ndata = 1)
    specconfig = SpecConfig(winsize = 1024,
                            noverlap = 512,
                            window = Windows.hanning(1024),
                            scaled = (a, fs) -> melscale(a, 1024; fs=fs),
                            preprocess_augment = x -> identity(x),
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
    @test hz2mel(60, true) ≈ 92.6819 atol=1e-3
    @test hz2mel.([110,220,440]) ≈ [1.65,3.3,6.6]
    @test hz2mel.([110,220,440], true) ≈ [164.4892, 308.0, 549.6387] atol=1e-3
    @test mel2hz(3) ≈ 200
    @test mel2hz(3, true) ≈ 1.8658 atol=1e-3
    @test mel2hz.([1,2,3,4,5]) ≈ [66.667,133.333,200.,266.667,333.333] atol=1e-3
    @test mel2hz.([1,2,3,4,5], true) ≈ [0.6214, 1.2433, 1.8658, 2.4889, 3.1125] atol=1e-3
    @test fft_frequencies(22050, 16) == [0.,1378.125,2756.25,4134.375,
                                         5512.5,6890.625,8268.75,9646.875,11025.]
    @test mel_frequencies(40, 0, 11025) ≈ [      0.,    85.3173,   170.6345,   255.9518,
                                            341.2690,  426.5863,   511.9035,   597.2208,
                                            682.5380,  767.8553,   853.1726,   938.4898,
                                           1024.8555, 1119.1141,  1222.0418,  1334.4360,
                                           1457.1675, 1591.1878,  1737.5322,  1897.3374,
                                           2071.8403, 2262.3926,  2470.4705,  2697.6858,
                                           2945.7988, 3216.7312,  3512.5820,  3835.6430,
                                           4188.4167, 4573.6358,  4994.2846,  5453.6214,
                                           5955.2046, 6502.9197,  7101.0094,  7754.1070,
                                           8467.2717, 9246.0280, 10096.4081, 11025.0   ] atol=1e-2
    @test mel_frequencies(40, 0, 11025, true) ≈ [   0.    ,   52.4593,   108.8501,   169.4668,
                                                  234.6263,  304.6690,   379.9608,   460.8952,
                                                  547.8949,  641.4145,   741.9427,   850.0046,
                                                  966.1649, 1091.0305,  1225.2537,  1369.5359,
                                                 1524.6309, 1691.3490,  1870.5612,  2063.2040,
                                                 2270.2838, 2492.8825,  2732.1632,  2989.3761,
                                                 3265.8650, 3563.0745,  3882.5574,  4225.9830,
                                                 4595.1456, 4991.9739,  5418.5413,  5877.0765,
                                                 6369.9751, 6899.8125,  7469.3570,  8081.5842,
                                                 8739.6929, 9447.1215, 10207.5662, 11025.    ] atol=1e-2
    
    @test getmelfilters(4800, 8, 4) ≈ [0. 0.00102068 0.         0.         0.;
                                       0. 0.00166109 0.         0.         0.;
                                       0. 0.         0.00187853 0.         0.;
                                       0. 0.         0.00024229 0.00123142 0.] atol=1e-3
end

@testset "embeddings" begin
    
    tsconfig = TSConfig(winsize = 4800, 
                        randsegment = false, 
                        preprocess_augment = x -> identity(x),
                        nchannels = 1,
                        ndata = 1)
    specconfig = SpecConfig(winsize = 1024,
                            noverlap = 512,
                            window = Windows.hanning(1024),
                            scaled = (a, fs) -> melscale(a, 1024; fs=fs),
                            preprocess_augment = x -> identity(x),
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
    x1 = randn(n)
    x2 = signal(x1, 9600)
    for x ∈ [x1, x2]
        @test apply(Amplify(Uniform(1.999999,2.000001)), x) ≈ 2 .* x atol=1e-3
        @test apply(PolarityInverse(), x) == -x
        @test apply(CircularShift(Binomial(1,1)), x) == circshift(x, 1)
        @test apply(TimeStretch(0.00000001), x) ≈ x atol=0.1
        @test apply(PitchShift(0.00000001), x) ≈ x atol=0.1
        @test std(apply(BackgroundNoise(), x) - x) ≈ √2 atol=0.2
        @test std(apply(BackgroundNoise(0), x) - x) ≈ √2 atol=0.2

        @test random_apply(PolarityInverse(), x; p=0) == x
        @test random_apply(PolarityInverse(), x; p=1) == -x
    end

end
