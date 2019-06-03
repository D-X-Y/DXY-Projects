require 'hdf5'
rapidjson = require 'rapidjson'

function hdf5_save(path, obj)
	local h = hdf5.open(path, 'w')
	local function r(prefix, o)
		for k, v in pairs(o) do
			local p = prefix..'/'..k
			if torch.isTypeOf(v, torch.CudaTensor) then
				h:write(p, v:float())
			elseif torch.isTensor(v) then
				h:write(p, v)
			elseif type(v) == 'number' then
				h:write(p, torch.DoubleTensor(1):fill(v))
			elseif type(v) == 'string' then
				h:write(p, torch.CharTensor(torch.CharStorage():string(v)))
			elseif type(v) == 'boolean' then
				h:write(p, torch.IntTensor(1):fill(v and 1 or 0))
			else
				r(p, v)
			end
		end
	end
	r('', obj)
	h:close()
end

function hdf5_load(path, fields)
	local res = {}

	local h = hdf5.open(path, 'r')
	if fields then
		local returnValue = false
		if type(fields) ~= 'table' then
			returnValue = true
			fields = {fields}
		end
		for _, f in ipairs(fields) do
			if not pcall(function()	res[f] = h:read('/'..f):all() end) then
				res[f] = nil
			end
		end
		if returnValue then
			res = res[fields[1]]
		end
	else
		res = h:all()
	end
	h:close()

	local function dfs(obj)
		for k, v in pairs(obj) do
			if tonumber(k) ~= nil then
				obj[k] = nil
				k = tonumber(k)
				obj[k] = v
			end

			if torch.isTypeOf(v, torch.CharTensor) or torch.isTypeOf(v, torch.ByteTensor) then
				obj[k] = v:storage():string()
			elseif torch.isTypeOf(v, torch.DoubleTensor) and v:nElement() == 1 then
				obj[k] = v:squeeze()
			elseif  torch.isTypeOf(v, torch.IntTensor) and v:nElement() == 1 and (v:squeeze() == 0 or v:squeeze() == 1) then
				obj[k] = v:squeeze() == 1 and true or false
			elseif type(v) == 'table' then
				dfs(v)
			end
		end
	end

	if type(res) == 'table' then
		dfs(res)
	end

	return res
end

json_load = rapidjson.load
json_save = function(path, obj)	rapidjson.dump(obj, path, {pretty = true, sort_keys = true}) end

function area_1(box)
	return (box[3] - box[1] + 1) * (box[4] - box[2] + 1)
end

function overlap(box1, box2)
	if torch.isTensor(box2) and box2:dim() == 2 then
		local res = box2.new(box2:size(1))
		for i = 1, res:nElement() do
			res[i] = overlap(box1, box2[i])
		end
		return res
	end

	local a1 = area_1(box1)
	local a2 = area_1(box2)

	local xx1 = math.max(box1[1], box2[1])
	local yy1 = math.max(box1[2], box2[2])
	local xx2 = math.min(box1[3], box2[3])
	local yy2 = math.min(box1[4], box2[4])

	local w = math.max(0.0, xx2 - xx1 + 1)
	local h = math.max(0.0, yy2 - yy1 + 1)
	local inter = w * h
	
	local ovr = inter / (a1 + a2 - inter)
	return ovr
end

function localizeMaxBox3d(scores, rois)
	if torch.isTensor(scores) and torch.isTensor(rois) then
		assert(scores:dim() == 3) -- numSamples x numClasses x numRois
		assert(rois:dim() == 3) -- numSamples x numRois x 4

		return rois:gather(2, ({scores:max(3)})[2]:expand(scores:size(1), scores:size(2), rois:size(3)))
	else
		assert(#scores == #rois)
		local res = torch.FloatTensor(#scores, scores[1]:size(1), 4)
		for exampleIdx = 1, res:size(1) do
			res[exampleIdx]:copy(rois[exampleIdx]:gather(1, ({scores[exampleIdx]:max(2)})[2]:expand(scores[exampleIdx]:size(1), rois[exampleIdx]:size(rois[exampleIdx]:dim()))))
		end
		return res
	end
end

function corloc(dataset_subset, localizedBoxes, classLabelInd)
	return mIOU(dataset_subset, localizedBoxes, 0.5, classLabelInd)
end

function mIOU(dataset_subset, localizedBoxes, corlocThreshold, classLabelInd)
	if type(localizedBoxes) == 'table' then
		localizedBoxes = localizeMaxBox3d(unpack(localizedBoxes))
	end
	assert(localizedBoxes:dim() == 3 and localizedBoxes:size(3) == 4)
	local beg_classLabelInd = classLabelInd == nil and 1 or classLabelInd
	local end_classLabelInd = classLabelInd == nil and localizedBoxes:size(2) or classLabelInd

	local mIOUs = {}
	for classLabelInd = beg_classLabelInd, end_classLabelInd  do
		local overlaps = {}
		for exampleIdx = 1, localizedBoxes:size(1) do
			local gtBoxes_ = dataset_subset:getGroundTruthBoxes(exampleIdx)
			local gtInds = gtBoxes_:select(2, 1):eq(classLabelInd):nonzero()
			if gtInds:nElement() > 0 then
				local gtBoxes = gtBoxes_:index(1, gtInds:squeeze(2)):narrow(2, 2, 4)
				local localizedBox = localizedBoxes[exampleIdx][classLabelInd]
				local maxOverlap = 0
				for i = 1, gtBoxes:size(1) do
					local o = overlap(gtBoxes[i], localizedBox)
					if corlocThreshold then
						o = o > corlocThreshold and 1 or 0
					end
					maxOverlap = math.max(maxOverlap, o)
				end
				table.insert(overlaps, maxOverlap)
			end
		end

		table.insert(mIOUs, torch.FloatTensor(#overlaps == 0 and {0.0} or overlaps):mean())
	end
	return torch.FloatTensor(mIOUs):mean()
end

function nms_mask(boxes, scores, overlap_threshold, score_threshold)
	local function nmsEx(boxes, scores, mask)
		--https://raw.githubusercontent.com/fmassa/object-detection.torch/master/nms.lua
		local xx1, yy1, xx2, yy2, w, h, area = boxes.new(), boxes.new(), boxes.new(), boxes.new(), boxes.new(), boxes.new(), boxes.new()
		local pick = torch.LongTensor()
		for classLabelInd = 1, scores:size(1) do
			local x1, y1, x2, y2 = boxes:select(2, 1), boxes:select(2, 2), boxes:select(2, 3), boxes:select(2, 4)
			area:cmul(x2 - x1 + 1, y2 - y1 + 1)
			pick:resize(area:size()):zero()

			local _, I = scores[classLabelInd]:sort(1)
			local overTh = scores[classLabelInd]:index(1, I):ge(score_threshold)
			if overTh:any() then
				I = I[overTh]
			else
				I:resize(0)
			end

			local count = 1
			while I:numel() > 0 do 
				local last = I:size(1)
				local i = I[last]

				pick[count] = i
				count = count + 1

				if last == 1 then
					break
				end

				I = I[{{1, last-1}}]

				xx1:index(x1, 1, I)
				yy1:index(y1, 1, I)
				xx2:index(x2, 1, I)
				yy2:index(y2, 1, I)

				xx1:cmax(x1[i])
				yy1:cmax(y1[i])
				xx2:cmin(x2[i])
				yy2:cmin(y2[i])

				w:add(xx2, -1, xx1):add(1):cmax(0)
				h:add(yy2, -1, yy1):add(1):cmax(0)
				
				local intersection = w:cmul(h)
				local IoU = h

				xx1:index(area, 1, I)
				IoU:cdiv(intersection, xx1 + area[i] - intersection)

				I = I[IoU:le(overlap_threshold)]
			end
			
			if count >= 2 then
				mask[classLabelInd]:scatter(1, pick[{{1, count-1}}], 1)
			end
		end
	end

	local mask = {}

	local threads = require 'threads'
	threads.Threads.serialization('threads.sharedserialize')
	local jobQueue = threads.Threads(16)
	for exampleIdx = 1, #scores do
		mask[exampleIdx] = torch.ByteTensor(scores[exampleIdx]:size()):zero()
		jobQueue:addjob(nmsEx, function() end, boxes[exampleIdx], scores[exampleIdx], mask[exampleIdx])
	end

	jobQueue:synchronize()

	return mask
end
