--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The full pre-activation ResNet variation from the technical report
-- "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027)
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local Replicate = nn.Replicate
local Squeeze = nn.Squeeze
local UpSample = nn.SpatialUpSamplingBilinear

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels
   local iSpatial

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   local function basicblock(n, stride, type)
      local nInputPlane = iChannels
      iChannels = n

      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))

      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
   end

   local function acc_until(nIn, nOut, kernel, stride, pad, name)

      local convs = nn.Sequential()
      convs:add(Convolution(nIn, nOut, kernel, kernel, stride, stride, pad, pad))
      local acc   = nn.Sequential()
      local grid_size = opt.grid_size

      if stride == 2 or grid_size == 1 then
	 acc:add(Convolution(nIn, 1, 1, 1, stride, stride, 0, 0))
	 grid_size = -1
      elseif stride == 1 then
	 if opt.grid_size == 0 then
            grid_size = iSpatial
         end 
	 acc:add(Max(grid_size, grid_size, grid_size, grid_size))
	 acc:add(Convolution(nIn, 1, 1, 1, 1, 1, 0, 0))
	 acc:add(UpSample(grid_size))
      else
	error('invalid dataset: ' .. opt.dataset)
      end
      acc:add(SBatchNorm(1))

      print_str = ('For lccl until [%s] input size : %2dx%2d'):format(name, iSpatial, iSpatial)
      print (print_str .. (' with grid %2d'):format(grid_size))

      local temp  = ReLU(true)
      temp.name = name
      temp.in_channel = nIn
      temp.out_channel = nOut
      temp.kernel_ori = kernel * kernel
      temp.kernel_acc = 1
      temp.stride = stride

      acc:add(temp)
      acc:add(Replicate(nOut, 2))
      acc:add(Squeeze())
      return nn.Sequential()
                :add(nn.ConcatTable()
                    :add(convs)
                    :add(acc))
                :add(nn.CMulTable(true))
   end


   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock_acc(n, stride, base_name, type)

      local nInputPlane = iChannels
      iChannels = n

      local block = nn.Sequential()
      local s = nn.Sequential()
      if type == 'both_preact' then
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
      elseif type ~= 'no_preact' then
         s:add(SBatchNorm(nInputPlane))
         s:add(ReLU(true))
      end
      s:add(acc_until(nInputPlane,n,3,stride,1, base_name .. '.acc_1'))

      iSpatial = iSpatial / stride

      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(acc_until(n,n,3,1,1, base_name .. '.acc_2'))

      return block
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))

   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride, base_name, type)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      s:add(block(features, stride, ('%s.%02d'):format(base_name, 1),
                  type == 'first' and 'no_preact' or 'both_preact'))
      for i=2,count do
         s:add(block(features, 1, ('%s.%02d'):format(base_name, i)))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
         [200] = {{3, 24, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64,  def[1], 1, 'first'))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))
      model:add(layer(block, 512, def[4], 2))
      model:add(ShareGradInput(SBatchNorm(iChannels), 'last'))
      model:add(ReLU(true))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 1000))
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      iSpatial = 32
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))     -- size 32x32
      model:add(layer(basicblock_acc, 16, n, 1, 'stage1'))   -- size 32x32
      model:add(layer(basicblock_acc, 32, n, 2, 'stage2'))   -- size 16x16
      model:add(layer(basicblock_acc, 64, n, 2, 'stage3'))   -- size 8 x 8
      model:add(ShareGradInput(SBatchNorm(iChannels), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      iSpatial = 32
      print(' | ResNet-' .. depth .. ' CIFAR-100')

      -- The ResNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(layer(basicblock_acc, 16, n, 1, 'stage1'))
      model:add(layer(basicblock_acc, 32, n, 2, 'stage2'))
      model:add(layer(basicblock_acc, 64, n, 2, 'stage3'))
      model:add(ShareGradInput(SBatchNorm(iChannels), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         if v.affine == true then
            v.weight:fill(1)
            v.bias:zero()
         end
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
