function branch_transform_rois_share_fc_layers(base_model, transformer)
	return nn.Sequential():
		add(RectangularRingRoiPooling(base_model.pooled_height, base_model.pooled_width, base_model.spatial_scale, base_model.spp_correction_params, transformer)):
		add(base_model.fc_layers_view(RoiReshaper)):
		add(share_weight_bias(base_model.fc_layers))
end

function RectangularRing(rois, scale_inner, scale_outer)
	local center_x = (rois:select(rois:dim(), 1) + rois:select(rois:dim(), 3)) / 2
	local center_y = (rois:select(rois:dim(), 2) + rois:select(rois:dim(), 4)) / 2
	local w_half = (rois:select(rois:dim(), 3) - rois:select(rois:dim(), 1)) / 2
	local h_half = (rois:select(rois:dim(), 4) - rois:select(rois:dim(), 2)) / 2
	
	rois:select(rois:dim(), 1):copy(center_x - w_half*scale_outer)
	rois:select(rois:dim(), 2):copy(center_y - h_half*scale_outer)
	rois:select(rois:dim(), 3):copy(center_x + w_half*scale_outer)
	rois:select(rois:dim(), 4):copy(center_y + h_half*scale_outer)
	rois:select(rois:dim(), 5):copy(center_x - w_half*scale_inner)
	rois:select(rois:dim(), 6):copy(center_y - h_half*scale_inner)
	rois:select(rois:dim(), 7):copy(center_x + w_half*scale_inner)
	rois:select(rois:dim(), 8):copy(center_y + h_half*scale_inner)
end

function MakeRectangularRingTransform(scale_inner, scale_outer)
	return function(rois) RectangularRing(rois, scale_inner, scale_outer) end
end

function BoxOriginal(rois)
end

CentralRegion1 = MakeRectangularRingTransform(0.0, 0.5)
CentralRegion2 = MakeRectangularRingTransform(0.3, 0.8)
BorderRegion1 = MakeRectangularRingTransform(0.5, 1.0)
BorderRegion2 = MakeRectangularRingTransform(0.8, 1.5)
ContextRegion = MakeRectangularRingTransform(1.0, opts.ROI_FACTOR)
BoxOriginal_ring = MakeRectangularRingTransform(1.0 / opts.ROI_FACTOR, 1.0)
ContextRegion_overlap = MakeRectangularRingTransform(0.8, 0.8 * opts.ROI_FACTOR)
ContextRegion_outer = MakeRectangularRingTransform(1.2, 1.2 * opts.ROI_FACTOR)
ContextRegion_big = MakeRectangularRingTransform(1.5, 2.0)
CentralRegion_big = MakeRectangularRingTransform(0.0, 2.0)
BoxScaleUp = MakeRectangularRingTransform(0.0, opts.ROI_FACTOR)

function BoxHalfLeft(rois)
	rois:select(rois:dim(), 3):add(rois:select(rois:dim(), 1)):div(2)
end

function BoxHalfRight(rois)
	rois:select(rois:dim(), 1):add(rois:select(rois:dim(), 3)):div(2)
end

function BoxHalfUp(rois)
	rois:select(rois:dim(), 4):add(rois:select(rois:dim(), 2)):div(2)
end

function BoxHalfBottom(rois)
	rois:select(rois:dim(), 2):add(rois:select(rois:dim(), 4)):div(2)
end

function DoubleUp(rois)
	rois:select(rois:dim(), 2):csub(rois:select(rois:dim(), 4) - rois:select(rois:dim(), 2))
end

function DoubleDown(rois)
	rois:select(rois:dim(), 4):add(rois:select(rois:dim(), 4) - rois:select(rois:dim(), 2))
end

function DoubleLeft(rois)
	rois:select(rois:dim(), 1):csub(rois:select(rois:dim(), 3) - rois:select(rois:dim(), 1))
end

function DoubleRight(rois)
	rois:select(rois:dim(), 3):add(rois:select(rois:dim(), 3) - rois:select(rois:dim(), 1))
end

function ShiftUp(rois)
	DoubleUp(rois)
	rois:select(rois:dim(), 4):add(rois:select(rois:dim(), 2)):div(2)
end

function ShiftDown(rois)
	DoubleDown(rois)
	rois:select(rois:dim(), 2):add(rois:select(rois:dim(), 4)):div(2)
end

function ShiftLeft(rois)
	DoubleLeft(rois)
	rois:select(rois:dim(), 3):add(rois:select(rois:dim(), 1)):div(2)
end

function ShiftRight(rois)
	DoubleRight(rois)
	rois:select(rois:dim(), 1):add(rois:select(rois:dim(), 3)):div(2)
end
