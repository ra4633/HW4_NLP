import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
BOS_TOKEN = "<extra_id_0>"

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        #TODO=> done
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.decoder_bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.examples = self.process_data(data_folder, split, self.tokenizer)  

    def process_data(self, data_folder, split, tokenizer):
        #TODO=> undergooing (maybe add some text normalization here)
        input_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        input_lines = [f'translate English to SQL: {line.strip()}' for line in input_lines]
        tokenized_input = tokenizer(input_lines, add_special_tokens=True, truncation=True)['input_ids']
        examples = []
        if split == 'test':
            for encoder_ids in tokenized_input:
                examples.append(
                    (
                        torch.tensor(encoder_ids, dtype=torch.long),
                        torch.tensor([self.decoder_bos_id], dtype=torch.long),
                    )
                )
            return examples
        output_lines = load_lines( os.path.join(data_folder, f'{split}.sql'))
        tokenized_output = tokenizer(output_lines, add_special_tokens=True, truncation=True)['input_ids']
        for encoder_ids, decoder_ids in zip(tokenized_input, tokenized_output):
            decoder_ids = [self.decoder_bos_id] + decoder_ids + [tokenizer.eos_token_id]
            examples.append(
                (
                    torch.tensor(encoder_ids, dtype=torch.long),
                    torch.tensor(decoder_ids, dtype=torch.long),
                )
            )
        return examples
        
    
    def __len__(self):
        #TODO=> done
        return len(self.examples)

    def __getitem__(self, idx):
        # TODO=> done
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO=>underdevlopment    
    encoder_ids = [item[0] for item in batch]
    decoder_full = [item[1] for item in batch]

    encoder_ids_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids_padded != PAD_IDX).long()
    decoder_full_padded = pad_sequence(decoder_full, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = decoder_full_padded[:, :-1]
    decoder_targets = decoder_full_padded[:, 1:]
    initial_decoder_inputs = decoder_full_padded[:, :1]
    return encoder_ids_padded, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs    

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO=> underdevelopment
    encoder_ids = [item[0] for item in batch]
    encoder_ids_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids_padded != PAD_IDX).long()
    initial_decoder_inputs =  pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    return encoder_ids_padded, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO+>done
    train_x = load_lines(os.path.join(data_folder, 'train.nl')) 
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x