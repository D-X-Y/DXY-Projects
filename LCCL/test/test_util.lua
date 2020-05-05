nn = require 'nn'
UpSample = nn.SpatialUpSamplingBilinear
Max = nn.SpatialMaxPooling
SBatchNorm = nn.SpatialBatchNormalization


model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 1, 5, 5))
model:add(UpSample(5))

-- FPROP a multi-resolution sample
input = torch.rand(3,5,5)
input:float()
model:forward(input)
-- Print the size of the Threshold outputs
list = model:listModules()
for i = 1, #list do
  print(list[i].output:size())
end
