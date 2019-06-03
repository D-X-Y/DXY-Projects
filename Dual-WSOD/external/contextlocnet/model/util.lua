require 'cudnn'

dofile('model/rectangularringroipooling.lua')
dofile('model/HingeCriterion.lua')
dofile('model/roi_transforms.lua')

local function module_typename(module)
	return torch.typename(module):sub(4)
end

function model_load(path, opts)
	local loaded = paths.extname(path) == 'lua' and {model_path = path} or hdf5_load(path)
	local opts = opts or loaded.meta.opts
	local model_definition = io.open(loaded.model_path or loaded.meta.model_path):read('*all')

	base_model = dofile(paths.concat('model', opts.BASE_MODEL .. '.lua'))(opts.PATHS.BASE_MODEL_CACHED)
	assert(loadstring(model_definition))()
	
	local function dfs(module, prefix)
		if module.weight then
			assert(loaded.parameters[prefix .. '_weight'] ~= nil)
			module.weight:copy(loaded.parameters[prefix .. '_weight'])
		end
		if module.bias then
			assert(loaded.parameters[prefix .. '_bias'] ~= nil)
			module.bias:copy(loaded.parameters[prefix .. '_bias'])
		end

		for i, submodule in ipairs(module.modules or {}) do
			dfs(submodule, (submodule.name and submodule.name[1]) or ((prefix or module_typename(module)) .. '_' .. module_typename(submodule) .. i))
		end
	end

	if loaded.parameters then
		dfs(model)
	end
	
	return loaded
end

function model_save(path, model, meta, epoch, log)
	local saved = {
		meta = meta,
		epoch = epoch,
		log = log,
		parameters = {}
	}

	local function dfs(module, prefix)
		if module.weight then
			local tensor_name = prefix .. '_weight'
			assert(saved.parameters[tensor_name] == nil or saved.parameters[tensor_name]:isSetTo(module.weight), torch.typename(module) .. ', ' ..prefix)
			saved.parameters[tensor_name] = module.weight
		end

		if module.bias then
			local tensor_name = prefix .. '_bias'
			assert(saved.parameters[tensor_name] == nil or saved.parameters[tensor_name]:isSetTo(module.bias), torch.typename(module) .. ', ' ..prefix)
			saved.parameters[tensor_name] = module.bias
		end

		for i, submodule in ipairs(module.modules or {}) do
			dfs(submodule, (submodule.name and submodule.name[1]) or ((prefix or module_typename(module)) .. '_' .. module_typename(submodule) .. i))
		end
	end

	dfs(model)

	hdf5_save(path, saved)
end

RoiReshaper = {
	inputSize = nil,

	StoreShape = function(this)
		local module = nn.Identity()
		function module:updateOutput(input)
			this.inputSize = input:size()
			return nn.Identity.updateOutput(self, input)
		end
		return module	
	end,

	RestoreShape = function(self, singletonDimension)
		return singletonDimension and DynamicView(function() return {-1, assert(self.inputSize)[2], numClasses, 1} end) or DynamicView(function() return {-1, assert(self.inputSize)[2], numClasses} end)
	end
}

function DynamicView(sizeFactory)
	local module = nn.View(-1)
	module.updateOutput = function(self, input) return nn.View.updateOutput(self:resetSize(unpack(sizeFactory())), input) end
	return module
end

function flatdim2(tensor)
	return tensor:contiguous():view(-1, unpack(torch.LongTensor(tensor:size()):sub(3, #tensor:size()):totable()))
end

function meandim2(tensor, batchSize)
	return tensor:contiguous():view(batchSize, -1, unpack(torch.LongTensor(tensor:size()):sub(2, #tensor:size()):totable())):mean(2):squeeze(2)
end

function share_weight_bias(module)
	return module:clone('weight', 'bias', 'gradWeight', 'gradBias')
end

function nn.Module.named(self, name)
	if not self.name then
		self.name = name
	else
		self.name = type(self.name) == 'table' and self.name or {self.name}
		table.insert(self.name, name)
	end
	return self
end

local nn_Module_findModules = nn.Module.findModules
function nn.Module.findModules(self, typename, container)
	for _, name in ipairs(type(self.name) == 'table' and self.name or (type(self.name) == 'string' and {self.name} or {})) do
		if name == typename then
			return {self}, {self}
		end
	end
	return nn_Module_findModules(self, typename, container)
end

function Probe(module, name, recursive)
	name = name or module_typename(module)
	if recursive and module.modules then
		for i = 1, #module.modules do
			module.modules[i] = Probe(module.modules[i], module.modules[i].name or (name .. '->' .. i), recursive)
		end
	end

	local module_updateOutput, module_updateGradInput, module_accGradParameters = module.updateOutput, module.updateGradInput, module.accGradParameters
	local fmtSize = function(tensor) return torch.isTensor(tensor) and ('('..('%d '):rep(tensor:dim())..')'):format(unpack(torch.LongTensor(tensor:size()):totable())) or tostring(#tensor)  end
	function module:updateOutput(input)
		print(name, 'updateOutput: in', '#input = ', fmtSize(input))
		local elapsed = gpuTicToc(function() self.output = module_updateOutput(self, input) end)
		print(name, 'updateOutput: out', ('%.4f ms'):format(elapsed*1000))
		return self.output
	end
	function module:updateGradInput(input, gradOutput)
		print(name, 'updateGradInput: in')
		local elapsed = gpuTicToc(function() self.gradInput = module_updateGradInput(self, input, gradOutput) end)
		print(name, 'updateGradInput: out', ('%.4f ms'):format(elapsed*1000))
		return self.gradInput
	end
	function module:accGradParameters(input, gradOutput, scale)
		print(name, 'accGradParameters: in')
		local elapsed = gpuTicToc(function() module_accGradParameters(self, input, gradOutput, scale) end)
		print(name, 'accGradParameters: out', ('%.4f ms'):format(elapsed*1000))
	end
	return module
end

function gpuTicToc(f)
	cutorch.synchronize()
	local tic = torch.tic()
	f()
	cutorch.synchronize()
	return torch.toc(tic)
end


collectgarbage()
