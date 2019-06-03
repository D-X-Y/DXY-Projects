dofile('opts.lua')
dofile('util.lua')
dofile('dataset.lua')
dofile('model/util.lua')

require 'optim'
dofile('fbnn_Optim.lua')

assert(os.getenv('CUDA_VISIBLE_DEVICES') ~= nil and cutorch.getDeviceCount() <= 1, 'SHOULD RUN ON ONE GPU FOR NOW')

torch.manualSeed(opts.SEED)
cutorch.manualSeedAll(opts.SEED)

example_loader_options_preset = {
	training = {
		numRoisPerImage = 8192,
		subset = 'trainval',
		hflips = true,
		numScales = 5,
	},
	evaluate = {
		numRoisPerImage = 8192,
		subset = 'trainval',
		hflips = true,
		numScales = 1,
	}
}

if paths.extname(opts.PATHS.MODEL) == 'lua' then
	loaded = model_load(opts.PATHS.MODEL, opts)
	meta = {
		model_path = loaded.model_path,
		opts = opts,
		example_loader_options = example_loader_options_preset
	}
	log = {{meta = meta}}
else
	loaded = model_load(opts.PATHS.MODEL)
	meta = loaded.meta
	log = loaded.log
	previous_epoch = loaded.epoch
end

batch_loader = ParallelBatchLoader(ExampleLoader(dataset, base_model.normalization_params, opts.IMAGE_SCALES, meta.example_loader_options)):setBatchSize({training = 1, evaluate = 1})

print(meta)

assert(model):cuda()
assert(criterion):cuda()
collectgarbage()

model:apply(function (x) x.for_each = x.apply end)
optimizer = nn.Optim(model, optimState)
optimalg = optim.sgd

for epoch = (previous_epoch or 0) + 1, opts.NUM_EPOCHS do
	if epoch > optimState_annealed.epoch then
		optimizer:setParameters(optimState_annealed)
	end

	batch_loader:training()
	model:training()
	for batchIdx = 1, batch_loader:getNumBatches() -1 do
		tic = torch.tic()

		scale_batches = batch_loader:forward()[1]
		scale0_rois = scale_batches[1][2]
		batch_images, batch_rois, batch_labels = unpack(scale_batches[2])
		batch_images_gpu = (batch_images_gpu or torch.CudaTensor()):resize(batch_images:size()):copy(batch_images)
		batch_labels_gpu = (batch_labels_gpu or torch.CudaTensor()):resize(batch_labels:size()):copy(batch_labels)

		cost = optimizer:optimize(optimalg, {batch_images_gpu, batch_rois}, batch_labels_gpu, criterion)

		collectgarbage()
		print('epoch', epoch, 'batch', batchIdx, cost, 'img/sec', batch_images:size(1) / torch.toc(tic))
	end

	if epoch % 5 == 0 or epoch == opts.NUM_EPOCHS or epoch == 1 then
		batch_loader:evaluate()
		model:evaluate()
		scores, labels, rois, costs, outputs, corlocs = {}, {}, {}, {}, {}, {}
		for batchIdx = 1, batch_loader:getNumBatches() - 1 do
			tic = torch.tic()

			scale_batches = batch_loader:forward()[1]
			scale0_rois = scale_batches[1][2]
			scale_outputs, scale_scores, scale_costs = {}, {}, {}
			for i = 2, #scale_batches do
				batch_images, batch_rois, batch_labels = unpack(scale_batches[i])
				batch_images_gpu = (batch_images_gpu or torch.CudaTensor()):resize(batch_images:size()):copy(batch_images)
				batch_labels_gpu = (batch_labels_gpu or torch.CudaTensor()):resize(batch_labels:size()):copy(batch_labels)

				batch_scores = model:forward({batch_images_gpu, batch_rois})

				cost = criterion:forward(batch_scores, batch_labels_gpu)
				
				table.insert(scale_scores, (type(batch_scores) == 'table' and batch_scores[1] or batch_scores):float())
				table.insert(scale_costs, cost)
				for _, output_field in ipairs(opts.OUTPUT_FIELDS) do
					module = model:findModules(output_field)[1]
					if module then
						scale_outputs[output_field] = scale_outputs[output_field] or {}
						table.insert(scale_outputs[output_field], module.output:transpose(2, 3):float())
					end
				end
			end

			for output_field, output in pairs(scale_outputs) do
				outputs[output_field] = outputs[output_field] or {}
				table.insert(outputs[output_field], torch.cat(output, 1):mean(1)[1])
			end

			table.insert(costs, torch.FloatTensor(scale_costs):mean())
			table.insert(scores, torch.cat(scale_scores, 1):mean(1))
			table.insert(labels, batch_labels:clone())
			table.insert(rois, scale0_rois:narrow(scale0_rois:dim(), 1, 4):clone()[1])
			
			collectgarbage()
			print('val', 'epoch', epoch, 'batch', batchIdx, costs[#costs], 'img/sec', (#scale_batches - 1) / torch.toc(tic))
		end

		for output_field, output in pairs(outputs) do
			corlocs[output_field] = corloc(dataset[batch_loader.example_loader:getSubset(batch_loader.train)], {output, rois})
		end

		table.insert(log, {
			training = false,
			epoch = epoch,
			mAP = dataset_tools.meanAP(torch.cat(scores, 1), torch.cat(labels, 1)),
			corlocs = corlocs,
			valCost = torch.FloatTensor(costs):mean(),
		})
	end

	if epoch % 5 == 0 or epoch == opts.NUM_EPOCHS then
		model:clearState()
		model_save(opts.PATHS.CHECKPOINT_PATTERN:format(epoch), model, meta, epoch, log)
	end

	json_save(opts.PATHS.LOG, log)
	io.stderr:write('log in "', opts.PATHS.LOG, '"\n')
end

