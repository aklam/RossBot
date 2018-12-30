import sys
sys.path.append('../nmt-keras')
sys.path.append('../nmt-keras/nmt_keras')
from config import load_parameters
from model_zoo import TranslationModel
import utils 
from keras_wrapper.cnn_model import loadModel
from data_engine.prepare_data import build_dataset, update_dataset_from_file


params = load_parameters()
params['DATA_ROOT_PATH'] = 'data/'
params['SRC_LAN'] = 'query'
params['TRG_LAN'] = 'reply'
params['TEXT_FILES'] = {'train': 'Cornell_train.', 'val': 'Cornell_valid.'}
params['TOKENIZATION_METHOD'] = 'tokenize_basic'

ds = build_dataset(params)

#params['INPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
#params['OUTPUT_VOCABULARY_SIZE'] = ds.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

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

training_params = {'n_epochs': 1, 'batch_size': 80,'maxlen': 30, 'epochs_for_save': 1, 'verbose': 1, 'eval_on_sets': [], 'reload_epoch': 0, 'epoch_offset': 0}

nmt_model.trainNet(ds, training_params)

