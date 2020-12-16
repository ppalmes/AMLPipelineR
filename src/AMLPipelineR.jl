module AMLPipelineR

using AMLPipelineBase
using AMLPipelineBase: fit!,transform!,fit_transform!
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: Machine, Learner, Transformer, Workflow, Computer
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

import AMLPipelineBase.AbsTypes: fit!, transform!

export fit!, transform!, fit_transform!
export Machine, Learner, Transformer, Workflow, Computer
export Baseline, Identity
export Imputer,OneHotEncoder,Wrapper
export PrunedTree,RandomForest,Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector
export CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline @pipelinex, @pipelinez
export Pipeline, ComboPipeline


include("rmachine.jl")
using .RMachine
export CRTLearner

end #module
