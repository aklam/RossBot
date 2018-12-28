import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')
from config import load_parameters
from model_zoo import TranslationModel
import utils 
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset

ds_round1 = loadDataset('query_to_reply/Dataset_Cornell_base.pkl')
params = load_parameters()
params['INPUT_VOCABULARY_SIZE'] = ds_round1.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = ds_round1.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

params['SOURCE_TEXT_EMBEDDING_SIZE'] = 300
params['TARGET_TEXT_EMBEDDING_SIZE'] = 300

params['SRC_PRETRAINED_VECTORS'] = '../Google_w2v.npy'
params['TRG_PRETRAINED_VECTORS'] = '../Google_w2v.npy'


#Model parameters
params['ENCODER_RNN_TYPE'] = 'GRU'
params['DECODER_RNN_TYPE'] = 'GRU'

#params['N_LAYERS_ENCODER'] = 2
#params['N_LAYERS_DECODER'] = 2
params['ENCODER_HIDDEN_SIZE'] = 512
params['DECODER_HIDDEN_SIZE'] = 512
params['MODEL_SIZE'] = 512
params['SRC_PRETRAINED_VECTORS_TRAINABLE'] = False
params['TRG_PRETRAINED_VECTORS_TRAINABLE'] = False 

#params['SKIP_VECTORS_HIDDEN_SIZE'] = 512
params['ATTENTION_SIZE'] = 512

nmt_model = TranslationModel(params, 
	model_type='GroundHogModel',
	model_name='Dec_27',
	vocabularies=ds_round1.vocabulary,
	store_path='trained_models/Dec_27/',
	verbose=True)

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = ds_round1.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source
nmt_model.setInputsMapping(inputMapping)

outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = ds_round1.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)

training_params = {'n_epochs': 1, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 1, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 0, 'epoch_offset': 0}

nmt_model.trainNet(ds_round1, training_params)


print("------- CORNELL FINISHED ROUND 1 -------")

ds_round2 = loadDataset('query_to_reply/Dataset_Cornell_Rd2.pkl')

inputMapping = dict()
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
    pos_source = ds_round2.ids_inputs.index(id_in)
    id_dest = nmt_model.ids_inputs[i]
    inputMapping[id_dest] = pos_source
nmt_model.setInputsMapping(inputMapping)

outputMapping = dict()
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
    pos_target = ds_round2.ids_outputs.index(id_out)
    id_dest = nmt_model.ids_outputs[i]
    outputMapping[id_dest] = pos_target
nmt_model.setOutputsMapping(outputMapping)

training_params_2 = {'n_epochs': 1, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 1, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 0, 'epoch_offset': 1}

nmt_model.trainNet(ds_round2, training_params_2)

