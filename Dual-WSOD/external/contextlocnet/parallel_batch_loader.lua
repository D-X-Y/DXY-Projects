--if nThreads = 0 do everything on main thread

require 'nn'

local ParallelBatchLoader, parent = torch.class('ParallelBatchLoader', 'nn.Module')

function ParallelBatchLoader:__init(example_loader, nThreads)
	parent.__init(self)

	self.example_loader = example_loader
	self.nThreads = nThreads or 16

	self.nextBatchIdx = 1
	self.preloadedBatchIdx = nil
	
	self.batchSize = {[true] = nil, [false] = nil}
	self.batchBuffers = nil
	self.currentBufferIdx = 1
	
	local threads = require 'threads'
	threads.Threads.serialization('threads.sharedserialize')
	self.jobQueue = threads.Threads(self.nThreads)

	parent:evaluate()
end

function ParallelBatchLoader:loadBatch(exampleIdxBegin)
	self.jobQueue:synchronize()

	self.currentBufferIdx = 3 - self.currentBufferIdx
	local batchTable = self.batchBuffers[self.currentBufferIdx]
	local isTrainingPhase = self.train

	for exampleIndexInBatch = 1, self:getBatchSize() do
		local exampleIdx = isTrainingPhase and torch.random(1, self:getNumExamples()) or (exampleIdxBegin - 1 + exampleIndexInBatch)
		local fillBatchTable = self.example_loader:loadExample(exampleIdx, isTrainingPhase)
		self.jobQueue:addjob(function()	fillBatchTable(exampleIndexInBatch, batchTable) end)
	end
end

function ParallelBatchLoader:getBatch(batchIdx)
	batchIdx = batchIdx or 1
	assert(batchIdx <= self:getNumBatches())
	
	local exampleIdxBegin = 1 + (batchIdx - 1) * self:getBatchSize()
	local exampleIdxEnd = 1 + math.min(batchIdx * self:getBatchSize(), self:getNumExamples())
	local effectiveBatchSize = exampleIdxEnd - exampleIdxBegin
	local oldBatchSize = self:getBatchSize()

	if batchIdx ~= self.preloadedBatchIdx or effectiveBatchSize ~= self:getBatchSize() then
		self:setBatchSize(effectiveBatchSize)
		self.preloadedBatchIdx = batchIdx
		self:loadBatch(exampleIdxBegin)
	end

	self.jobQueue:synchronize()
	local loadedBatchTable = self.batchBuffers[self.currentBufferIdx]

	if self:getBatchSize() ~= oldBatchSize then
		self:setBatchSize(oldBatchSize)
	end

	local nextBatchIdx = batchIdx + 1
	if nextBatchIdx < self:getNumBatches() then
		self.preloadedBatchIdx = nextBatchIdx
		self:loadBatch(exampleIdxBegin + self:getBatchSize())
	end

	return loadedBatchTable
end

function ParallelBatchLoader:updateOutput()
	assert(self:getBatchSize())
	assert(self.nextBatchIdx)
	self.output = self:getBatch(self.nextBatchIdx)
	self.nextBatchIdx = self.nextBatchIdx + 1
	return self.output
end

function ParallelBatchLoader:setBatchSize(batchSize)
	if type(batchSize) == 'table' then
		self.batchSize = {[true] = batchSize.training, [false] = batchSize.evaluate}
	else
		self.batchSize[self.train] = batchSize
		if self.batchSize[not self.train] == nil then
			self.batchSize[not self.train] = batchSize
		end
	end

	self:reinitBatchBuffers()

	return self
end

function ParallelBatchLoader:reinitBatchBuffers()
	self.batchBuffers = {self.example_loader:makeBatchTable(self:getBatchSize(), self.train), self.example_loader:makeBatchTable(self:getBatchSize(), self.train)}
end

function ParallelBatchLoader:getBatchSize()
	return self.batchSize[self.train]
end

function ParallelBatchLoader:getNumBatches()
	return torch.ceil(self:getNumExamples() / self:getBatchSize())
end

function ParallelBatchLoader:getNumExamples()
	return self.example_loader:getNumExamples(self.train)
end

function ParallelBatchLoader:training()
	parent:training()
	self.nextBatchIdx = 1
	self:reinitBatchBuffers()
end

function ParallelBatchLoader:evaluate()
	parent:evaluate()
	self.nextBatchIdx = 1
	self:reinitBatchBuffers()
end
