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
    dwt_matrix,
    idwt_matrix,
    predict
export wavelet, WT

abstract type AbstractWaveletTransform end

for transformtype in [:MODWT, :DWT, :WPT]
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

function dwt_matrix(X::AbstractVecOrMat, trans::DWT)
    return hcat([dwt(X[:, i], trans.wavelet, trans.nlevels) for i in 1:size(X, 2)]...)
end

function idwt_matrix(Xw::AbstractArray, trans::DWT)
    n, nvars = size(Xw)
    hcat([idwt(Xw[:, i], trans.wavelet, trans.nlevels) for i in 1:nvars]...)
end

function wavelm(X::AbstractVecOrMat, Y::AbstractVecOrMat, trans::DWT)
    Xw = dwt_matrix(X, trans)
    Yw = dwt_matrix(Y, trans)
    n = size(Xw, 1)
    indices = [detailrange(n, j) for j in 1:trans.nlevels]
    push!(indices, 1:detailindex(n, trans.nlevels, 1)-1) # scaling coef indices
    BB = [Xw[ii, :] \ Yw[ii, :] for ii in indices]
    n = size(X, 1)
    mx = div(length(X), n)
    my = div(length(Y), n)
    return WaveletRegressionModel(X, Y, n, mx, my, Xw, Yw, trans, BB)
end


end # module
