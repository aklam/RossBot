import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')

import utils 
from config import load_parameters
from data_engine.prepare_data import keep_n_captions
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.utils import decode_predictions_beam_search
from model_zoo import TranslationModel

params = load_parameters()
dataset = loadDataset('query_to_reply/Dataset_Cornell_base.pkl')

dataset.setInput('data/Ross_test.query',
	'test', 
	type='text',
	id='source_text',
	pad_on_batch=True,
	tokenization='tokenize_basic',
	fill='end',
	max_text_len=100,
	min_occ=0)

dataset.setInput(None,
	'test',
	type='ghost',
	id='state_below',
	required=False)


## get model predictions  
params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

Control_model = loadModel('trained_models/Control_M3', 18)

params_prediction = {'max_batch_size': 50, 'predict_on_sets': ['test'], 'beam_size': 12, 'maxlen': 50, 'model_inputs': ['source_text', 'state_below'], 'model_outputs': ['target_text'], 'dataset_inputs': ['source_text', 'state_below'], 'dataset_outputs': ['target_text'], 'normalize': True, 'alpha_factor': 0.6 }

Control_predictions = Control_model.predictBeamSearchNet(dataset, params_prediction)['test']


vocab = dataset.vocabulary['target_text']['idx2words']
Control_predictions = decode_predictions_beam_search(Control_predictions, vocab, verbose=params['VERBOSE'])


## see how they compare to ground truth
#
from keras_wrapper.extra.read_write import list2file
from keras_wrapper.extra import evaluation

Control_path = 'Control_M2.pred'
list2file(Control_path, Control_predictions)


dataset.setOutput('data/Ross_test.reply', 
	'test', 
	type='text', 
	id='target_text', 
	pad_on_batch=True, 
	tokenization='tokenize_basic', 
	sample_weights=True, 
	max_text_len=30, 
	max_words=0)

keep_n_captions(dataset, repeat=1, n=1, set_names=['test'])

metric = 'coco'
#Apply sampling
extra_vars = dict()
extra_vars['tokenize_f'] = eval('dataset.' + 'tokenize_basic')
extra_vars['language'] = params['TRG_LAN']
extra_vars['test'] = dict()
extra_vars['test']['references'] = dataset.extra_variables['test']['target_text']
Control_metrics = evaluation.select[metric](pred_list=Control_predictions, verbose=1, extra_vars=extra_vars, split='test')


print("Control:")
print(Control_metrics)

