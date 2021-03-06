module SepdQuantile

using Distributions, Random, LinearAlgebra, StatsBase, SpecialFunctions, ProgressMeter, ForwardDiff, StatsModels, RCall, PDMats
using DataFrames, QuantileRegressions

import Distributions: pdf, _logpdf, sampler, _rand!, logpdf, @check_args, dim
import Base: rand

include("Distributions/aepd.jl")
include("Estimation/mcmc.jl")
include("Estimation/quantConvert.jl")
include("Estimation/frequentist.jl")


export mcmc, Sampler, acceptance, Aepd, mcτ, quantfreq, quantconvert

end
