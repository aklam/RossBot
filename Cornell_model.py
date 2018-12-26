import sys
sys.path.append('../nmt-keras')
from data_engine.prepare_data import keep_n_captions
from keras_wrapper.dataset import Dataset, saveDataset

dataset_name = 'Cornell_base'
cornell_path = '/data'

ds = Dataset(dataset_name, cornell_path)

ds.setInput('data/Cornell_train_query.en',
	'train',
	type='text',
	id='source_text',
	pad_on_batch=True,
	tokenization='tokenize_none',
	build_vocabulary=True,
	fill='end',
	max_text_len=30,
	max_words=30000,
	min_occ=0)

ds.setInput('data/Cornell_valid_query.en',
	'val',
	type='text',
	id='source_text',
	pad_on_batch=True,
	tokenization='tokenize_none',
	fill='end',
	max_text_len=30,
	max_words=0)

ds.setInput('data/Cornell_train_reply_offset.en',
	'train',
	type='text',
	id='state_below',
	required=False,
	tokenization='tokenize_none',
	pad_on_batch=True,
	build_vocabulary='target_text',
	offset=1,
	fill='end',
	max_text_len=30,
	max_words=30000)

ds.setInput(None, 
	'val',
	type='ghost',
	id='state_below',
	pad_on_batch=True,
	required=False)

ds.setOutput('data/Cornell_train_reply.en',
	'train',
	type='text',
	id='target_text',
	tokenization='tokenize_none',
	build_vocabulary=True,
	pad_on_batch=True,
	sample_weights=True,
	max_text_len=30,
	max_words=30000,
	min_occ=0)

ds.setOutput('data/Cornell_valid_reply.en',
	'val',
	type='text',
	id='target_text',
	pad_on_batch=True,
	tokenization='tokenize_none',
	sample_weights=True,
	max_text_len=30,
	max_words=0)

ds.merge_vocabularies(['source_text', 'target_text'])

print(ds)

keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

saveDataset(ds, 'query_to_reply')

sys.path.append('../nmt-keras/nmt_keras')
from config import load_parameters
from model_zoo import TranslationModel
import utils 
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
params = load_parameters()
params['INPUT_VOCABULARY_SIZE'] = ds.vocabulary_len['source_text']
params['OUTPUT_VOCABULARY_SIZE'] = ds.vocabulary_len['target_text']
params['MODEL_NAME'] = "Test123"
params['STORE_PATH'] = "~/RossBot/Test123"

#params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
#params['TARGET_TEXT_EMBEDDING_SIZE'] = 300

#params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
#params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'


#Model parameters
params['REBUILD_DATASET']  =  False 
params['ENCODER_RNN_TYPE'] = 'GRU'
params['DECODER_RNN_TYPE'] = 'GRU'

#params['N_LAYERS_ENCODER'] = 2
#params['N_LAYERS_DECODER'] = 2
params['ENCODER_HIDDEN_SIZE'] = 512
params['DECODER_HIDDEN_SIZE'] = 512
params['MODEL_SIZE'] = 512

params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
params['ATTENTION_SIZE'] = 512

nmt_model = TranslationModel(params, 
	model_type='GroundHogModel',
	model_name='Dec_26',
	vocabularies=ds.vocabulary,
	store_path='trained_models/Dec_26/',
	verbose=True)

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = ds.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source

nmt_model.setInputsMapping(inputMapping)
outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = ds.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)

training_params = {'n_epochs': 100, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 20, 'verbose': 0, 'eval_on_sets': [], 'n_parallel_loaders': 8, 'reload_epoch': 0}

nmt_model.trainNet(ds, training_params)

