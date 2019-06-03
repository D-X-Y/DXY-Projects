return function(modelPath)
	local vggf = torch.load(modelPath)

	local conv_layers = nn.Sequential()
	for i = 1, 14 do
		conv_layers:add(vggf:get(i))
	end

	local fc_layers = nn.Sequential()
	for i = 17, 22 do
		fc_layers:add(vggf:get(i))
	end

	return {
		conv_layers = conv_layers, 
		fc_layers = fc_layers, 
		channel_order = 'bgr', 
		spatial_scale = 1 / 16, 
		fc_layers_output_size = 4096,
		pooled_height = 6, 
		pooled_width = 6, 
		spp_correction_params = {offset0 = -18, offset = 0.0},
		--spp_correction_params = {offset0 = -18.0, offset = 9.5},
		fc_layers_view = function(RoiReshaper) return nn.View(-1):setNumInputDims(3) end,
		normalization_params = {channel_order = 'bgr', rgb_mean = {122.7717, 115.9465, 102.9801}, rgb_std = nil, scale = 255}
	}
end
