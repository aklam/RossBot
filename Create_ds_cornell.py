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

#ds.merge_vocabularies(['source_text', 'target_text'])

print(ds)

keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

saveDataset(ds, 'query_to_reply')

