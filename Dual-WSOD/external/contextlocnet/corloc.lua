dofile('opts.lua')
dofile('util.lua')
dofile('dataset.lua')

opts.SCORES_FILES = #arg >= 1 and arg or {opts.PATHS.SCORES_PATTERN:format('trainval')}

loaded = hdf5_load(opts.SCORES_FILES[1], {'subset', 'rois', 'labels', 'output'})
outputs = {}

for i = 1, #opts.SCORES_FILES do
	outputs_i = hdf5_load(opts.SCORES_FILES[i], 'outputs')
	for output_field, scores in pairs(outputs_i) do
		outputs[output_field] = {}
		for exampleIdx = 1, #scores do
			outputs[output_field][exampleIdx] = (outputs[output_field][exampleIdx] or scores[exampleIdx]:clone():zero()):add(scores[exampleIdx]:div(#opts.SCORES_FILES))
		end
	end
end

res = {training_MAP = dataset_tools.meanAP(loaded.output, loaded.labels)}
for output_field, scores in pairs(outputs) do
	res[output_field] = {by_class = {}, _mean = corloc(dataset[loaded.subset], {scores, loaded.rois})}
	for classLabelInd, classLabel in ipairs(classLabels) do
		res[output_field].by_class[classLabels[classLabelInd]] = corloc(dataset[loaded.subset], {scores, loaded.rois}, classLabelInd)
	end
end

json_save(opts.PATHS.CORLOC, res)
print('result in ' .. opts.PATHS.CORLOC)
