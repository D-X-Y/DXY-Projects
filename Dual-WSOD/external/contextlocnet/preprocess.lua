require 'cudnn'
require 'loadcaffe'
require 'image'

matio = require 'matio'
voc_tools = dofile('pascal_voc.lua')

dofile('opts.lua')

function VGGF()
	local model_converted = loadcaffe.load(opts.PATHS.BASE_MODEL_RAW.PROTOTXT, opts.PATHS.BASE_MODEL_RAW.CAFFEMODEL, 'cudnn'):float()
	torch.save(opts.PATHS.BASE_MODEL_CACHED, model_converted)
end

function VOC()
	local function copy_proposals_in_dataset(trainval_test_mat_paths, voc)
		local subset_paths = {{'train', trainval_test_mat_paths.trainval}, {'val', trainval_test_mat_paths.trainval}, {'test', trainval_test_mat_paths.test}}

		local m = {train = {}, val = {}, test = {}}
		local b = {train = nil, val = nil, test = nil}
		local s = {train = nil, val = nil, test = nil}
		for _, t in ipairs(subset_paths) do
			local h = matio.load(t[2])
			b[t[1]] = h.boxes
			s[t[1]] = h.boxScores
			for exampleIdx = 1, #b[t[1]] do
				m[t[1]][h.images[exampleIdx]:storage():string()] = exampleIdx
			end
		end

		for _, subset in ipairs{'train', 'val', 'test'} do
			voc[subset].rois = {}
			for exampleIdx = 1, voc[subset]:getNumExamples() do
				local ind = m[subset][voc[subset]:getImageFileName(exampleIdx)]
				local box_scores = s[subset] and s[subset][ind] or torch.FloatTensor(b[subset][ind]:size(1), 1):zero()
				--local box_scores = torch.FloatTensor(b[subset][ind]:size(1), 1):zero()
				voc[subset].rois[exampleIdx] = torch.cat(b[subset][ind]:index(2, torch.LongTensor{2, 1, 4, 3}):float() - 1, box_scores)

				if s[subset] then
					voc[subset].rois[exampleIdx] = voc[subset].rois[exampleIdx]:index(1, ({box_scores:squeeze(2):sort(1, true)})[2]:sub(1, math.min(box_scores:size(1), 2048)))
				end
			end
			voc[subset].getProposals = function(self, exampleIdx)
				return self.rois[exampleIdx]
			end
		end

		voc['trainval'].getProposals = function(self, exampleIdx)
			return exampleIdx <= self.train:getNumExamples() and self.train:getProposals(exampleIdx) or self.val:getProposals(exampleIdx - self.train:getNumExamples())
		end
	end

	local function filter_proposals(voc)
		local min_width_height = 20
		for _, subset in ipairs{'train', 'val', 'test'} do
			for exampleIdx = 1, voc[subset]:getNumExamples() do
				local x1, y1, x2, y2 = unpack(voc[subset].rois[exampleIdx]:split(1, 2))
				local channels, height, width = unpack(image.decompressJPG(voc[subset]:getJpegBytes(exampleIdx)):size():totable())
				
				assert(x1:ge(0):all() and x1:le(width):all())
				assert(x2:ge(0):all() and x2:le(width):all())
				assert(y1:ge(0):all() and y1:le(height):all())
				assert(y2:ge(0):all() and y2:le(height):all())
				assert(x1:le(x2):all() and y1:le(y2):all())

				voc[subset].rois[exampleIdx] = voc[subset].rois[exampleIdx]:index(1, (x2 - x1):ge(min_width_height):cmul((y2 - y1):ge(min_width_height)):squeeze(2):nonzero():squeeze(2))
			end
		end
	end

	local voc = voc_tools.load(opts.PATHS.VOC_DEVKIT_VOCYEAR)
	copy_proposals_in_dataset(opts.PATHS.PROPOSALS, voc)
	filter_proposals(voc)
	torch.save(opts.PATHS.DATASET_CACHED, voc)
end

for _, a in ipairs(arg) do
	print('Preprocessing', a)
	_G[a]()
end
print('Done')
