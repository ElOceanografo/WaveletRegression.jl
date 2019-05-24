module WaveletRegression

using Wavelets
using Statistics
export
    wavelm,
    AbstractWaveletTransform,
    MODWT,
    DWT,
    DWPT,
    WaveletRegressionModel,
    modwt_matrix,
    imodwt_matrix,
    predict
export wavelet, WT

abstract type AbstractWaveletTransform end

for transformtype in [:MODWT, :DWT, :DWPT]
    @eval begin
        struct $transformtype <: AbstractWaveletTransform
            wavelet::DiscreteWavelet
            nlevels::Integer
        end
        Wavelets.wavelet(trans::$(transformtype)) = trans.wavelet
        nlevels(trans::$(transformtype)) = trans.nlevels
    end
end

struct WaveletRegressionModel
    X::AbstractArray
    Y::AbstractArray
    n::Integer
    mx::Integer
    my::Integer
    Xw::AbstractArray
    Yw::AbstractArray
    trans::AbstractWaveletTransform
    coefficients
end
Wavelets.wavelet(mod::WaveletRegressionModel) = wavelet(mod.trans)
nlevels(mod::WaveletRegressionModel) = nlevels(mod.trans)

function predict(mod::WaveletRegressionModel, newX)
    @assert size(newX) == size(mod.X)
    newXw = modwt_matrix(newX, mod.trans)
    Yw = cat([reshape(newXw[:, j, :] * mod.coefficients[j], (mod.n, 1, mod.my))
        for j in 1:nlevels(mod)+1]..., dims=2)
    Y = imodwt_matrix(Yw, mod.trans)
    return Y
end
predict(mod::WaveletRegressionModel) = predict(mod, mod.X)

function modwt_matrix(X::AbstractVecOrMat, trans::MODWT)
    return cat([modwt(X[:, i], trans.wavelet, trans.nlevels) for i in 1:size(X, 2)]..., dims=3)
end

function imodwt_matrix(Xw::AbstractArray, trans::MODWT)
    n, nlevels, nvars = size(Xw)
    hcat([imodwt(Xw[:, :, i], trans.wavelet) for i in 1:nvars]...)
end


function wavelm(X::AbstractVecOrMat, Y::AbstractVecOrMat, trans::MODWT)
    Xw = modwt_matrix(X, trans)
    Yw = modwt_matrix(Y, trans)
    BB = [Xw[:, j, :] \ Yw[:, j, :] for j in 1:(trans.nlevels+1)]
    n = size(X, 1)
    mx = div(length(X), n)
    my = div(length(Y), n)
    return WaveletRegressionModel(X, Y, n, mx, my, Xw, Yw, trans, BB)
end

end # module
