local DATA = os.getenv('DATA') or 'data'
local DATA_COMMON = os.getenv('DATA_COMMON') or paths.concat(DATA, 'common')

PATHS = 
{
	EXTERNAL = 
	{
		PRETRAINED_MODEL_VGGF = 
		{
			PROTOTXT = paths.concat(DATA_COMMON, 'VGG_CNN_F_deploy.prototxt'),
			CAFFEMODEL = paths.concat(DATA_COMMON, 'VGG_CNN_F.caffemodel'),
		},

		SSW_VOC2007 =
		{
			trainval = paths.concat(DATA_COMMON, 'SelectiveSearchVOC2007trainval.mat'),
			test = paths.concat(DATA_COMMON, 'SelectiveSearchVOC2007test.mat')
		},

		SSW_VOC2012 =
		{
			trainval = paths.concat(DATA_COMMON, 'selective_search_data/voc_2012_trainval.mat'),
			test = paths.concat(DATA_COMMON, 'selective_search_data/voc_2012_test.mat')
		},
		
		VOC_DEVKIT_VOCYEAR =
		{
			VOC2007 = paths.concat(DATA_COMMON, 'VOCdevkit_2007/VOC2007'),
			VOC2012 = paths.concat(DATA_COMMON, 'VOCdevkit_2012/VOC2012')
		}
	},
	
	BASE_MODEL_CACHED = 
	{
		VGGF = paths.concat(DATA_COMMON, 'VGG_CNN_F.t7')
	},

	DATASET_CACHED_PATTERN = paths.concat(DATA_COMMON, '%s_%s.t7'),
	CHECKPOINT_PATTERN = paths.concat(DATA, 'model_epoch%02d.h5'),
	LOG = paths.concat(DATA, 'log.json'),
	SCORES_PATTERN = paths.concat(DATA, 'scores_%s.h5'),
	CORLOC = paths.concat(DATA, 'corloc.json'),
	DETECTION_MAP = paths.concat(DATA, 'detection_mAP.json'),
}

local DATASET = os.getenv('DATASET') or 'VOC2007'
local NUM_EPOCHS = tonumber(os.getenv('NUM_EPOCHS')) or 30
local SUBSET = os.getenv('SUBSET') or 'trainval'
local BASE_MODEL = 'VGGF'

opts = {
	ROI_FACTOR = 1.8,
	SEED = 1,
	
	NMS_OVERLAP_THRESHOLD = 0.4,
	NMS_SCORE_THRESHOLD = 1e-4,
	
	IMAGE_SCALES = {{608, 800}, {496, 656}, {400, 544}, {720, 960}, {864, 1152}}, --{{608, 800}, {368, 480}, {432, 576}, {528, 688}, {656, 864}, {912, 1200}}

	NUM_SCALES = 5,
	NUM_EPOCHS = NUM_EPOCHS,
	
	OUTPUT_FIELDS = {'output_prod'},
	DATASET = DATASET,
	BASE_MODEL = BASE_MODEL,

	SUBSET = SUBSET,
	PATHS = 
	{
		MODEL = arg[1],

		DATA = DATA,
		DATA_COMMON = DATA_COMMON,

		CHECKPOINT_PATTERN = PATHS.CHECKPOINT_PATTERN,
		LOG = PATHS.LOG,
		SCORES_PATTERN = PATHS.SCORES_PATTERN,

		BASE_MODEL_CACHED = PATHS.BASE_MODEL_CACHED[BASE_MODEL],
		BASE_MODEL_RAW = PATHS.EXTERNAL['PRETRAINED_MODEL_' .. BASE_MODEL],
		
		PROPOSALS = PATHS.EXTERNAL['SSW_' .. DATASET],
		
		VOC_DEVKIT_VOCYEAR = PATHS.EXTERNAL.VOC_DEVKIT_VOCYEAR[DATASET],
		DATASET_CACHED = PATHS.DATASET_CACHED_PATTERN:format(DATASET, 'SSW'),

		CORLOC = PATHS.CORLOC,
		DETECTION_MAP = PATHS.DETECTION_MAP,
		RUN_STATS_PATTERN = PATHS.RUN_STATS_PATTERN
	}
}
