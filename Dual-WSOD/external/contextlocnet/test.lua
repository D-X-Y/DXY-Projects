dofile('opts.lua')
dofile('util.lua')
dofile('dataset.lua')
dofile('model/util.lua')

assert(os.getenv('CUDA_VISIBLE_DEVICES') ~= nil and cutorch.getDeviceCount() <= 1, 'SHOULD RUN ON ONE GPU FOR NOW')

loaded = model_load(opts.PATHS.MODEL, opts)

meta = {
	opts = opts,
	training_meta = loaded.meta,
	example_loader_options = {
		evaluate = {
			numRoisPerImage = 8192,
			subset = opts.SUBSET,
			hflips = true,
			numScales = opts.NUM_SCALES
		}
	}
}

batch_loader = ParallelBatchLoader(ExampleLoader(dataset, base_model.normalization_params, opts.IMAGE_SCALES, meta.example_loader_options)):setBatchSize({evaluate = 1})

print(meta)
assert(model):cuda()
assert(criterion):cuda()
collectgarbage()

tic_start = torch.tic()

batch_loader:evaluate()
model:evaluate()
scores, labels, rois, outputs = {}, {}, {}, {}
for batchIdx = 1, batch_loader:getNumBatches() do
	tic = torch.tic()

	scale_batches = batch_loader:forward()[1]
	scale0_rois = scale_batches[1][2]
	scale_outputs, scale_scores, scale_costs = {}, {}, {}
	for i = 2, #scale_batches do
		batch_images, batch_rois, batch_labels = unpack(scale_batches[i])
		batch_images_gpu = (batch_images_gpu or torch.CudaTensor()):resize(batch_images:size()):copy(batch_images)
		batch_labels_gpu = (batch_labels_gpu or torch.CudaTensor()):resize(batch_labels:size()):copy(batch_labels)
		if nn.gModule then
			batch_scores = model:forward({batch_images_gpu, batch_rois, scale0_rois})
		else
			batch_scores = model:forward({batch_images_gpu, batch_rois})
		end

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
		table.insert(outputs[output_field], torch.cat(output, 1):mean(1):squeeze(1))
	end

	table.insert(scores, torch.cat(scale_scores, 1):mean(1))
	table.insert(labels, batch_labels:clone())
	table.insert(rois, scale0_rois:narrow(scale0_rois:dim(), 1, 4):clone()[1])
	
	collectgarbage()
	print('val', 'batch', batchIdx, torch.FloatTensor(scale_costs):mean(), 'img/sec', (#scale_batches - 1) / torch.toc(tic))
end

subset = batch_loader.example_loader:getSubset(batch_loader.train)
hdf5_save(opts.PATHS.SCORES_PATTERN:format(subset), {
	subset = subset,
	meta = meta,

	rois = rois,
	labels = torch.cat(labels, 1),
	output = torch.cat(scores, 1),
	outputs = outputs,
})

print('DONE:', torch.toc(tic_start), 'sec')
