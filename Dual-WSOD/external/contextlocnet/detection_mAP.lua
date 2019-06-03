dofile('opts.lua')
dofile('util.lua')
dofile('dataset.lua')
threads = require 'threads'

local MATLAB = assert((#sys.execute('which matlab') > 0 and 'matlab -r') or (#sys.execute('which octave') > 0 and 'octave --eval'), 'matlab or octave not found in PATH')
local subset = 'test'
output_field = opts.OUTPUT_FIELDS[1]

opts.SCORES_FILES = #arg >= 1 and arg or {opts.PATHS.SCORES_PATTERN:format(subset)}
rois = hdf5_load(opts.SCORES_FILES[1], 'rois')

scores = {}
for i = 1, #opts.SCORES_FILES do
	scores_i = hdf5_load(opts.SCORES_FILES[i], 'outputs/' .. output_field)
	for exampleIdx = 1, #scores_i do
		scores[exampleIdx] = (scores[exampleIdx] or scores_i[exampleIdx]:clone():zero()):add(scores_i[exampleIdx]:div(#opts.SCORES_FILES))
	end
end

local detrespath = dataset_tools.package_submission(opts.PATHS.DATA, dataset, opts.DATASET, subset, 'comp4_det', rois, scores, nms_mask(rois, scores, opts.NMS_OVERLAP_THRESHOLD, opts.NMS_SCORE_THRESHOLD))
local opts = opts

if dataset[subset].objectBoxes == nil then
	print('detection mAP cannot be computed for ' .. opts.DATASET .. '. Quitting.')
	print(('VOC submission saved in "%s/results-%s-%s-%s.tar.gz"'):format(opts.PATHS.DATA, opts.DATASET, 'comp4_det', subset))
	os.exit(0)
end

res = {[output_field] = {_mean = nil, by_class = {}}}
APs = torch.FloatTensor(numClasses):zero()

local imgsetpath = paths.tmpname()
os.execute(('sed \'s/$/ -1/\' %s > %s'):format(paths.concat(opts.PATHS.VOC_DEVKIT_VOCYEAR, 'ImageSets', 'Main', subset .. '.txt'), imgsetpath)) -- hack for octave

jobQueue = threads.Threads(numClasses)
for classLabelInd, classLabel in ipairs(classLabels) do
	jobQueue:addjob(function()
		os.execute(('%s "oldpwd = pwd; cd(\'%s\'); addpath(fullfile(pwd, \'VOCcode\')); VOCinit; cd(oldpwd); VOCopts.testset = \'%s\'; VOCopts.detrespath = \'%s\'; VOCopts.imgsetpath = \'%s\'; classLabel = \'%s\'; [rec, prec, ap] = VOCevaldet(VOCopts, \'comp4\', classLabel, false); dlmwrite(sprintf(VOCopts.detrespath, \'resu4\', classLabel), ap); quit;"'):format(MATLAB, paths.dirname(opts.PATHS.VOC_DEVKIT_VOCYEAR), subset, detrespath, imgsetpath, classLabel))
		return tonumber(io.open(detrespath:format('resu4', classLabel)):read('*all'))
	end, function(ap) res[output_field].by_class[classLabel] = ap; APs[classLabelInd] = ap; end)
end
jobQueue:synchronize()
os.execute('[ -t 1 ] && reset')

res[output_field]._mean = APs:mean()

json_save(opts.PATHS.DETECTION_MAP, res)
print('result in ' .. opts.PATHS.DETECTION_MAP)
