module TestRMachine

using AMLPipelineR
using Test
using DataFrames
using RDatasets

function caretrun()
   crt = CRTLearner(Dict(:learner=>"rf",:fitControl=>"trainControl(method='cv')"))
   iris=dataset("datasets","iris")
   x=iris[:,1:4]  
   y=iris[:,5] |> Array{String}
   @test (fit_transform!(crt,x,y) .== y ) |> sum == 150
end
@testset "caret" begin
	caretrun()
end

end
