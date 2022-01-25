module SepdQuantile

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ProgressMeter, ForwardDiff
using DataFrames

import Distributions: pdf, _logpdf, sampler, _rand!,
    logpdf, @check_args
import Base: rand

include("Distributions/aepd.jl")
include("Estimation/mcmc.jl")

export mcmc, Sampler, acceptance, Aepd


end # module
