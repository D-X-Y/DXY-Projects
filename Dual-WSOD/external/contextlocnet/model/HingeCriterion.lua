local HingeCriterion, parent = torch.class('HingeCriterion', 'nn.Criterion')

function HingeCriterion:__init(margin)
	parent.__init(self)
	self.sizeAverage=true
	
	self.sequence=nn.Sequential()
	self.sequence:add(nn.CMulTable())
	self.sequence:add(nn.MulConstant(-1,true))
	self.sequence:add(nn.AddConstant(margin or 1, true))
	self.sequence:add(nn.ReLU(true))
	
	self.gradient=torch.Tensor()
end

function HingeCriterion:setFactor(factor)
	self.factor = factor
	return self
end

function HingeCriterion:updateOutput(input, target)
	self.sequence:forward({input,target})
	self.output=self.sequence.output:sum()
	local p = (self.sizeAverage and 1/input:size(1) or 1) * (self.factor or 1)
	self.output = self.output * p
	return self.output
end


function HingeCriterion:updateGradInput(input, target)
	local p = (self.sizeAverage and 1/input:size(1) or 1) * (self.factor or 1)

	self.gradient:resize(self.sequence.output:size()):fill(p)
	self.sequence:backward({input,target}, self.gradient)
	self.gradInput=self.sequence.gradInput[1]
	return self.gradInput
end

function HingeCriterion:type(type)
	parent.type(self, type)
	self.sequence:type(type)
	self.gradient:type(type)
	return self
end
