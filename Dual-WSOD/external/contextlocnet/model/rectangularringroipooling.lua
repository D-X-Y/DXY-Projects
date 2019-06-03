require 'cunn'
require 'libcucontextlocnet'

local RectangularRingRoiPooling, parent = torch.class('RectangularRingRoiPooling', 'nn.Module')

function RectangularRingRoiPooling:__init(pooled_height, pooled_width, spatial_scale, scale_correction_params, roi_pre_transformer)
	parent.__init(self)

	assert(pooled_height > 0, 'pooled_h must be > 0')
	assert(pooled_width > 0, 'pooled_w must be > 0');

	self.pooled_height = pooled_height
	self.pooled_width = pooled_width
	self.spatial_scale = spatial_scale or 1.0

	self.scale_correction_params = scale_correction_params
	self.roi_pre_transformer = roi_pre_transformer
end

function RectangularRingRoiPooling:preprocess_rois(raw_rois)
	for i = 1, raw_rois:size(1) do
		self.preprocessed_rois[i]:select(2, 1):fill(i - 1)
	end
	self.preprocessed_rois:narrow(self.preprocessed_rois:dim(), 2, 4):copy(raw_rois:narrow(raw_rois:dim(), 1, 4))
	local rois = self.preprocessed_rois:narrow(self.preprocessed_rois:dim(), 2, 8)
	
	if self.roi_pre_transformer then
		self.roi_pre_transformer(rois)
	end
	
	local offset0, offset, spatial_scale = self.scale_correction_params.offset0, self.scale_correction_params.offset, self.spatial_scale
	rois:select(rois:dim(), 1):add(offset0 + offset):mul(spatial_scale):add(0.5):floor()
	rois:select(rois:dim(), 2):add(offset0 + offset):mul(spatial_scale):add(0.5):floor()
	rois:select(rois:dim(), 3):add(offset0 - offset):mul(spatial_scale):add(-0.5):ceil()
	rois:select(rois:dim(), 4):add(offset0 - offset):mul(spatial_scale):add(-0.5):ceil()

	rois:select(rois:dim(), 5):add(offset0 + offset):mul(spatial_scale):add(0.5):floor()
	rois:select(rois:dim(), 6):add(offset0 + offset):mul(spatial_scale):add(0.5):floor()
	rois:select(rois:dim(), 7):add(offset0 - offset):mul(spatial_scale):add(-0.5):ceil()
	rois:select(rois:dim(), 8):add(offset0 - offset):mul(spatial_scale):add(-0.5):ceil()
end

function RectangularRingRoiPooling:updateOutput(input)
	self.preprocessed_rois = (self.preprocessed_rois or torch.CudaTensor()):resize(input[2]:size(1), input[2]:size(2), 1 + 8):zero()
	self:preprocess_rois(input[2])

	self.argmax = self.argmax or torch.CudaIntTensor()
	input[1].contextlocnet.updateOutput(self, input[1], self.preprocessed_rois)
	return self.output
end

function RectangularRingRoiPooling:updateGradInput(input, gradOutput)
	self.gradInput = type(self.gradInput) == 'table' and (self.gradInput[1]  or torch.CudaTensor()) or self.gradInput

	input[1].contextlocnet.updateGradInput(self, input[1], self.preprocessed_rois, gradOutput)
	self.rois_zero_grad = (self.rois_zero_grad or input[2].new()):resizeAs(input[2]):zero()
	self.gradInput = {self.gradInput, self.rois_zero_grad}
	return self.gradInput
end
