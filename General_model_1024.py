import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')
from config import load_parameters
from model_zoo import TranslationModel
import utils 
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset

ds = loadDataset('query_to_reply/Dataset_Cornell_base.pkl')
params = load_parameters()
params['INPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
params['TARGET_TEXT_EMBEDDING_SIZE'] = 300

params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'


#Model parameters
params['ENCODER_RNN_TYPE'] = 'GRU'
params['DECODER_RNN_TYPE'] = 'GRU'

params['N_LAYERS_ENCODER'] = 2
params['N_LAYERS_DECODER'] = 2
params['ENCODER_HIDDEN_SIZE'] = 1024
params['DECODER_HIDDEN_SIZE'] = 1024
params['MODEL_SIZE'] = 1024
params['SRC_PRETRAINED_VECTORS_TRAINABLE'] = False
params['TRG_PRETRAINED_VECTORS_TRAINABLE'] = False 

params['ATTENTION_SIZE'] = 1024

## Added reload to train epoch11
params['RELOAD'] = 10
params['RELOAD_EPOCH'] = True
params['REBUILD_DATASET'] = False
params['DATA_ROOT_PATH'] = 'data/'

nmt_model = TranslationModel(params, 
	model_type='GroundHogModel',
    weights_path='trained_models/1024_Base/epoch_10_init.h5',
	model_name='1024_Base_Epoch11',
	vocabularies=ds.vocabulary,
	store_path='trained_models/1024_Base_Epoch11/',
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

training_params = {'n_epochs': 11, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 1, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 10, 'epoch_offset': 10}

nmt_model.trainNet(ds, training_params)

