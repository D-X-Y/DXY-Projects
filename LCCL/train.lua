--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      if n % self.opt.print_freq == 0 or n == trainSize then
          print((' | Epoch: [%03d/%03d][%03d/%03d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
             epoch, self.opt.nEpochs, n, trainSize, timer:time().real, dataTime, loss, top1, top5))
     end

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local zeroS, totalS = 0.0, 1e-5 -- count sparsity
   local sparsity_layer = 0
   local N = 0
   local sparsity = self:init_sparsity()

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      if n % self.opt.print_freq == 0 or n == size then
          print((' | Test: [%03d/%03d][%03d/%03d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
                     epoch, self.opt.nEpochs, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))
      end

      timer:reset()
      dataTimer:reset()

      if epoch % self.opt.cal_sparsity == 0 or epoch == self.opt.nEpochs or self.opt.testOnly then
	 sparsity = self:count_sparsity(sparsity)
      end

   end
   self.model:training()
   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f   LearningRate: %.4f'):format(
      epoch, top1Sum / N, top5Sum / N, self.optimState.learningRate))

   if epoch % self.opt.cal_sparsity == 0 or epoch == self.opt.nEpochs or self.opt.testOnly then
     self:print_sparsity(sparsity)
   end

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:init_sparsity() -- count sparsity
   local sparsity = {}
   local list = self.model:listModules()
   local sparsity_layer = 0
   local str = 'acc'
   for i = 1,#list do
   if list[i].name and string.find(list[i].name, str) > 0 then
       sparsity_layer = sparsity_layer + 1
       current_l   = {}
       current_l.in_channel     = list[i].in_channel
       current_l.out_channel    = list[i].out_channel
       current_l.kernel_ori     = list[i].kernel_ori
       current_l.kernel_acc     = list[i].kernel_acc
       current_l.output_spatial = 0
       current_l.zero           = 0
       current_l.total          = 0
       current_l.idx            = sparsity_layer
       current_l.calculation    = 0
       current_l.zeros_cal      = 0
       current_l.saved_cal      = 0
       current_l.pre_zero       = 0
       current_l.pre_acc        = 0
       current_l.extra_cost     = 0
       sparsity[sparsity_layer] = current_l
   end
   end
   sparsity.sparsity_layer = sparsity_layer
   return sparsity
end

function Trainer:count_sparsity(sparsity) -- count sparsity
   local list = self.model:listModules()
   local str = 'acc'
   local sparsity_layer = 0
   for i = 1,#list do
   if list[i].name and string.find(list[i].name, str) > 0 then
      sparsity_layer = sparsity_layer + 1
      local output = list[i].output
      assert (output:size(2) == 1, 'accelerate layer\'s channel must be one')
      local current_l = sparsity[sparsity_layer]
      if current_l.output_spatial ~= output:size(3) * output:size(4) then
         current_l.output_spatial = output:size(3) * output:size(4)
      end
      output = output:storage()
      local zero, total = 0.0, #output
      for j = 1,#output do
          if output[j] == 0 then 
             zero = zero + 1
          end
      end
      local matrix_K = current_l.kernel_ori * current_l.in_channel
      local calculation = matrix_K * current_l.out_channel * current_l.output_spatial
      current_l.calculation = current_l.calculation + calculation
      current_l.zeros_cal   = current_l.zeros_cal + zero / total * calculation
      local conv_save       = zero / total * calculation
      local extra_cost      = 1/current_l.in_channel / (current_l.kernel_ori / current_l.kernel_acc)
      if self.opt.grid_size == 1 or list[i].stride == 2 then
        extra_cost = extra_cost * calculation
      elseif self.opt.grid_size > 1 then
	extra_cost = extra_cost * calculation / self.opt.grid_size / self.opt.grid_size 
	extra_cost = extra_cost + current_l.out_channel * current_l.output_spatial
	extra_cost = extra_cost + current_l.in_channel * current_l.output_spatial * list[i].stride * list[i].stride
      elseif self.opt.grid_size == 0 then
	grid_size = torch.sqrt(current_l.output_spatial)
	extra_cost = extra_cost * calculation / grid_size / grid_size 
	extra_cost = extra_cost + current_l.out_channel * current_l.output_spatial
	extra_cost = extra_cost + current_l.in_channel * current_l.output_spatial * list[i].stride * list[i].stride
      else
	error('unknoen grid size : ' .. self.opt.grid_size)
      end
		
      current_l.extra_cost  = current_l.extra_cost + extra_cost
      current_l.saved_cal   = current_l.saved_cal + conv_save - extra_cost
      current_l.pre_zero    = current_l.pre_zero + zero
      current_l.pre_acc     = current_l.pre_acc + total
      current_l.name        = list[i].name
      sparsity[sparsity_layer] = current_l
   end
   end
   return sparsity
end

function Trainer:print_sparsity(sparsity) -- count sparsity
   local list = self.model:listModules()
   local str = 'acc'
   local sparsity_layer = 0
   local zeros, saved, calA, final = 0.0, 0.0, 0.0, 0.0
   local pre_z, pre_t = 0.0, 0.0
   -- For those the same block zeros
   for i = 1,#list do
   if list[i].name and string.find(list[i].name, str) then
      sparsity_layer = sparsity_layer + 1
      if sparsity_layer % 2 == 0 then
        local aft = sparsity[sparsity_layer]
        local pre = sparsity[sparsity_layer-1]
	assert(string.find(aft.name, 'acc_2') and string.find(pre.name, 'acc_1'), 'fatal name :' .. pre.name)
	pre.final_save = pre.saved_cal
	aft.final_save = aft.saved_cal
	if pre.zeros_cal == pre.calculation or aft.zeros_cal == aft.calculation then
          pre.final_save = pre.calculation - pre.extra_cost
          aft.final_save = aft.calculation - aft.extra_cost
	end
      end
   end
   end

   sparsity_layer = 0
   for i = 1,#list do
   if list[i].name and string.find(list[i].name, str) then
      sparsity_layer = sparsity_layer + 1
      local current_l = sparsity[sparsity_layer]

      layer_str = ('[Layer-%02d : %s]'):format(sparsity_layer, current_l.name)
      print_str = (' [zero: %8.6f] [save: %6.3f] [final save: %6.3f] (cal: %9.1f)'):format(
                      current_l.zeros_cal / current_l.calculation, current_l.saved_cal / current_l.calculation, current_l.final_save / current_l.calculation, current_l.calculation/1000)
      output_size = torch.sqrt( current_l.output_spatial)
      print (layer_str .. print_str .. ('  %2dx%2d'):format(output_size,output_size) .. ('  last %4d nonzeros'):format(current_l.calculation - current_l.zeros_cal))
      zeros = zeros + current_l.zeros_cal
      saved = saved + current_l.saved_cal
      final = final + current_l.final_save
      calA  = calA  + current_l.calculation
      pre_z = pre_z + current_l.pre_zero
      pre_t = pre_t + current_l.pre_acc
   end
   end
   print (('Total , zero: %.3f, save: %.3f , final: %.3f ( %.3f )'):format(
        zeros / calA, saved / calA, final /calA, pre_z / pre_t))
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end

   if self.opt.netType == 'preresnet-acc' then
      decay = epoch >= math.ceil(self.opt.nEpochs*0.90) and 3 or 
              epoch >= math.ceil(self.opt.nEpochs*0.70) and 2 or 
              epoch >= math.ceil(self.opt.nEpochs*0.45) and 1 or 
              epoch >= math.ceil(self.opt.nEpochs*0.03) and 0 or 1
   end

   if self.opt.netType == 'preresnet-bottleneck-acc' then
      decay = epoch >= math.ceil(self.opt.nEpochs*0.90) and 3 or 
              epoch >= math.ceil(self.opt.nEpochs*0.70) and 2 or 
              epoch >= math.ceil(self.opt.nEpochs*0.45) and 1 or 
              epoch >= math.ceil(self.opt.nEpochs*0.03) and 0 or 1
   end

   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
