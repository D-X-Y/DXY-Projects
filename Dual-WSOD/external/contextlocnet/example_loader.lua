local ExampleLoader, parent = torch.class('ExampleLoader')

function ExampleLoader:__init(dataset, normalization_params, scales, example_loader_opts)
	self.scales = scales
	self.normalization_params = normalization_params
	self.example_loader_opts = example_loader_opts
	self.dataset = dataset
end

local function table2d(I, J, elem_generator)
	local res = {}
	for i = 1, I do
		res[i] = {}
		for j = 1, J do
			res[i][j] = elem_generator(i, j)
		end
	end
	return res
end

local function subtract_mean(dst, src, normalization_params)
	local channel_order = assert(({rgb = {1, 2, 3}, bgr = {3, 2, 1}})[normalization_params.channel_order])
	for c = 1, 3 do
		dst[c]:copy(src[channel_order[c]]):add(-normalization_params.rgb_mean[channel_order[c]])
		if normalization_params.rgb_std then
			dst[c]:div(normalization_params.rgb_std[channel_order[c]])
		end
	end
end

local function rescale(img, max_height, max_width)
	--local height_width = math.max(dhw_rgb:size(3), dhw_rgb:size(2))
	--local im_scale = target_height_width / height_width
	local scale_factor = max_height / img:size(2)
	if torch.round(img:size(3) * scale_factor) > max_width then
		scale_factor = math.min(scale_factor, max_width / img:size(3))
	end

	return image.scale(img, math.min(max_width, img:size(3) * scale_factor), math.min(max_height, img:size(2) * scale_factor))
end

local function flip(images_j, rois_j)
	image.hflip(images_j, images_j)
	rois_j:select(2, 1):mul(-1):add(images_j:size(3))
	rois_j:select(2, 3):mul(-1):add(images_j:size(3))

	local tmp = rois_j:select(2, 1):clone()
	rois_j:select(2, 1):copy(rois_j:select(2, 3))
	rois_j:select(2, 3):copy(tmp)
end

local function insert_dummy_dim1(...)
	for _, tensor in ipairs({...}) do
		tensor:resize(1, unpack(tensor:size():totable()))
	end
end

function ExampleLoader:makeBatchTable(batchSize, isTrainingPhase)
	local o = self:getPhaseOpts(isTrainingPhase)
	local num_jittered_copies = isTrainingPhase and 2 or (1 + (o.hflips and 2 or 1) * o.numScales)

	return table2d(batchSize, num_jittered_copies, function() return {torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()} end)
end

function ExampleLoader:loadExample(exampleIdx, isTrainingPhase)
	local o = self:getPhaseOpts(isTrainingPhase)
	
	local labels_loaded = self.dataset[o.subset]:getLabels(exampleIdx)
	local rois_loaded = self.dataset[o.subset]:getProposals(exampleIdx)
	local jpeg_loaded = self.dataset[o.subset]:getJpegBytes(exampleIdx)
	local scales = o.scales or self.scales
	local normalization_params = self.normalization_params

	local scale_inds = isTrainingPhase and {0, torch.random(1, o.numScales)} or torch.range(0, o.numScales):totable()
	local hflips = isTrainingPhase and (o.hflips and torch.random(0, 1) or 0) or (o.hflips and 2 or 0) -- 0 is no_flip, 1 is do_flip, 2 is both
	local rois_perm = isTrainingPhase and torch.randperm(rois_loaded:size(1)) or torch.range(1, rois_loaded:size(1))

	return function(indexInBatch, batchTable)
		image = image or require 'image'
		local img_original = image.decompressJPG(jpeg_loaded, 3, normalization_params.scale == 255 and 'byte' or 'float')
		local height_original, width_original = img_original:size(2), img_original:size(3)

		local rois_scale0 = rois_loaded:index(1, rois_perm:sub(1, math.min(rois_loaded:size(1), o.numRoisPerImage)):long())
		rois_scale0[1]:copy(torch.FloatTensor{0, 0, width_original - 1, height_original - 1, 0.0}:sub(1, rois_scale0:size(2)))

		for j, scale_ind in ipairs(scale_inds) do
			local images, rois, labels = unpack(batchTable[indexInBatch][j])

			local img_scaled = scale_ind == 0 and img_original:clone() or rescale(img_original, scales[scale_ind][1], scales[scale_ind][2])
			local width_scaled, height_scaled = img_scaled:size(3), img_scaled:size(2)

			subtract_mean(images:resize(img_scaled:size()), img_scaled, normalization_params)
			rois:cmul(rois_scale0, torch.FloatTensor{{width_scaled / width_original, height_scaled / height_original, width_scaled / width_original, height_scaled / height_original, 1.0}}:narrow(2, 1, rois_scale0:size(2)):contiguous():expandAs(rois_scale0))
			labels:resize(labels_loaded:size()):copy(labels_loaded)

			if hflips == 1 then
				flip(images, rois)
			elseif scale_ind ~= 0 and hflips == 2 then
				local jj = #batchTable[indexInBatch] - j + 2
				local images_flipped, rois_flipped, labels_flipped = unpack(batchTable[indexInBatch][jj])
				images_flipped:resizeAs(images):copy(images)
				rois_flipped:resizeAs(rois):copy(rois)
				labels_flipped:resizeAs(labels):copy(labels)
				flip(images_flipped, rois_flipped)
				insert_dummy_dim1(images_flipped, rois_flipped, labels_flipped)
			end

			insert_dummy_dim1(images, rois, labels)
		end

		collectgarbage()
	end
end

function ExampleLoader:getNumExamples(isTrainingPhase)
	return self.dataset[self:getSubset(isTrainingPhase)]:getNumExamples()
end

function ExampleLoader:getPhaseOpts(isTrainingPhase)
	return isTrainingPhase and self.example_loader_opts['training'] or self.example_loader_opts['evaluate']
end

function ExampleLoader:getSubset(isTrainingPhase)
	return self:getPhaseOpts(isTrainingPhase).subset
end
