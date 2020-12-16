module RMachine

using RCall
using DataFrames
using Random

using AMLPipelineBase
using AMLPipelineBase.AbsTypes
using AMLPipelineBase.Utils

import AMLPipelineBase.AbsTypes: fit!, transform!

const RLearners=["rf","svm"]

function __init__()
   #packages = ["caret","e1071","gam","randomForest",
   #            "nnet","kernlab","grid","MASS","pls"]
   rpackages = ["caret","e1071","gam","randomForest"]
   for pk in rpackages
      rcall(:library,pk,"lib=.libPaths()")
   end
end

export CRTLearner,fit!,transform!
# CARET wrapper that provides access to all learners.
# 
# Options for the specific CARET learner is to be passed
# in `options[:impl_options]` dictionary.
mutable struct CRTLearner <: Learner
   name
   model
   function CRTLearner(args::Dict{Symbol,<:Any} = Dict())
      fitControl="trainControl(method = 'cv',number = 5,repeats = 5)"
      fitControl="trainControl(method = 'none')"
      default_args = Dict{Symbol,Any}(
                      :name => "rlearner",
                      :learner => "rf",
                      :fitControl => fitControl,
                      :impl_args => Dict{Symbol,Any}()
                     )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      rl = cargs[:learner]
      if !(rl in RLearners)
         println("$rl is not supported.")
         println()
         throw(ArgumentError("Argument keyword error"))
      end
      new(cargs[:name],cargs)
   end
end

function fit!(crt::CRTLearner,x::DataFrame,y::Vector) 
   rmodel = rcall(:train,x,y,method=crt.model[:learner],trControl = reval(crt.model[:fitControl]))
   crt.model[:rmodel] = rmodel
end

function transform!(crt::CRTLearner,x::DataFrame) 
   res = rcall(:predict,crt.model[:rmodel],x)
   return rcopy(res) |> Array
end


end # module

