if opts.DATASET == 'VOC2007' or opts.DATASET == 'VOC2012' then
	dataset_tools = dofile('pascal_voc.lua')
	classLabels = dataset_tools.classLabels
	numClasses = dataset_tools.numClasses
end

dataset = torch.load(opts.PATHS.DATASET_CACHED)

dofile('parallel_batch_loader.lua')
dofile('example_loader.lua')
