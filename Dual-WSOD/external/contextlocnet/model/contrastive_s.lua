fc8r = nn.Linear(base_model.fc_layers_output_size, numClasses):named('fc8r')

model = nn.Sequential():
	add(nn.ParallelTable():
		add(base_model.conv_layers):
		add(RoiReshaper:StoreShape())
	):
	add(nn.ConcatTable():
		add(branch_transform_rois_share_fc_layers(base_model, BoxOriginal)):
		add(branch_transform_rois_share_fc_layers(base_model, BoxOriginal_ring)):
		add(branch_transform_rois_share_fc_layers(base_model, ContextRegion))
	):
	add(nn.ConcatTable():
		add(nn.Sequential():
			add(nn.SelectTable(1)):
			add(nn.Linear(base_model.fc_layers_output_size, numClasses):named('fc8c')):
			add(RoiReshaper:RestoreShape()):
			named('output_fc8c')
		):
		add(nn.Sequential():
			add(nn.ConcatTable():
				add(nn.Sequential():
					add(nn.SelectTable(2)):
					add(share_weight_bias(fc8r)):
					named('output_fc8d_origring')
				):
				add(nn.Sequential():
					add(nn.SelectTable(3)):
					add(share_weight_bias(fc8r)):
					add(nn.MulConstant(-1)):
					named('output_fc8d_context')
				)
			):
			add(nn.CAddTable()):
			add(RoiReshaper:RestoreShape(4)):
			add(cudnn.SpatialSoftMax()):
			add(nn.Squeeze(4)):
			named('output_softmax')
		)
	):
	add(nn.CMulTable():named('output_prod')):
	add(nn.Sum(2))

--classification_criterion = nn.BCECriterion(nil, false)
--classification_criterion.updateOutput = function(self, input, target) return nn.BCECriterion.updateOutput(self, input, target * 0.5 + 0.5) end
--classification_criterion.updateGradInput = function(self, input, target) return nn.BCECriterion.updateGradInput(self, input, target * 0.5 + 0.5) end
--criterion = classification_criterion

criterion = HingeCriterion():setFactor(1 / numClasses)
optimState = {learningRate = 5e-3, momentum = 0.9, weightDecay = 5e-4}
optimState_annealed = {learningRate = 5e-4, momentum = 0.9, weightDecay = 5e-4, epoch = 10}
