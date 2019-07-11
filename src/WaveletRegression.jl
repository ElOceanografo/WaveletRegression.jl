
module WaveletRegression

using Wavelets
using Statistics
using LinearAlgebra

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
    predict,
    predict_wavespace,
    coef,
    ncoeffs,
    dof,
    nlevels,
    residuals,
    regression_stderr,
    coef_stderr

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
coef(mod::WaveletRegressionModel) = mod.coefficients
ncoeffs(mod::WaveletRegressionModel) = sum(length(B) for B in mod.coefficients)
dof(mod::WaveletRegressionModel) = mod.n - ncoeffs(mod)

function predict_wavespace(mod::WaveletRegressionModel, newXw::AbstractArray)
    n = size(newXw, 1)
    Yw = cat([reshape(newXw[:, j, :] * mod.coefficients[j], (n, 1, mod.my))
        for j in 1:nlevels(mod)+1]..., dims=2)
    return Yw
end

function predict(mod::WaveletRegressionModel, newX::AbstractVecOrMat)
    @assert size(newX, 2) == size(mod.X, 2)
    newXw = modwt_matrix(newX, mod.trans)
    Yw = predict_wavespace(mod, newXw)
    Y = imodwt_matrix(Yw, mod.trans)
    return Y
end
predict(mod::WaveletRegressionModel) = predict(mod, mod.X)

function residuals(mod::WaveletRegressionModel)
    Yw_pred = predict_wavespace(mod, mod.Xw)
    return mod.Yw .- Yw_pred
end

function regression_stderr(mod::WaveletRegressionModel, unbiased=true)
    resid = residuals(mod)
    s2 = reshape(sum(abs2 , resid, dims=1) / dof(mod), (mod.my, :))
    if unbiased
        return s2
    else
        return s2 * dof(mod) / mod.n
    end
end

function coef_stderr(mod::WaveletRegressionModel, unbiased=true)
    # https://en.wikipedia.org/wiki/Ordinary_least_squares
    s2 = regression_stderr(mod, unbiased)
    L = mod.trans.nlevels + 1
    Qxx =  [mod.Xw[:, j, :]' * mod.Xw[:, j, :] for j in 1:L]
    SE = [similar(B) for B in coef(mod)]
    for j in 1:L
        for i in 1:mod.my
            SE[j][:, i] .= sqrt.(diag(s2[i, j] .* inv(Qxx[j])))
        end
    end
    return SE
end

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
