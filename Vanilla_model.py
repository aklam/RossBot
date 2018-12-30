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
params['INPUT_VOCABULARY_SIZE'] = 0 #ds.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
params['OUTPUT_VOCABULARY_SIZE'] = 0 #ds.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

nmt_model = TranslationModel(params, 
	model_type='GroundHogModel',
	model_name='Vanilla_model_resume_test',
	vocabularies=ds.vocabulary,
	store_path='trained_models/Vanilla_model_resume_test/',
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

training_params = {'n_epochs': 1, 'batch_size': 20,'maxlen': 30, 'epochs_for_save': 4, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 0, 'epoch_offset': 0}

nmt_model.trainNet(ds, training_params)

