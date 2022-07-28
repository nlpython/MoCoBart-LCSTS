from torch.utils.data import Dataset
from processing.tools import clean
from loguru import logger
from tqdm import tqdm
import os
import pickle
import json
import torch


class BartDataset(Dataset):

    def __init__(self, tokenizer, args, chunk, mode='train'):

        self.tokenizer = tokenizer
        self.data = self._load_and_cache_examples(args, chunk, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


    def _load_and_cache_examples(self, args, chunk, mode='train'):
        """
        Loads a data file into a list of `InputExample`s.
        Args:
             args: args
             mode: train or test
        :return: data
        """
        if mode == 'train':
            data_file = os.path.join(args.data_dir, 'PART_I.json')
            cache_file = os.path.join(args.data_dir, 'cache/train_set/train_{}.pkl'.format(chunk))
        elif mode == 'test':
            data_file = os.path.join(args.data_dir, 'PART_II.json')
            cache_file = os.path.join(args.data_dir, 'cache/eval_set/eval_{}.pkl'.format(chunk))
        else:
            raise ValueError('mode should be train or test')

        if os.path.exists(cache_file):
            logger.info("Loading examples from cache file {}".format(cache_file))
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data

        logger.info("Not found cache file, create examples to file {}".format(cache_file))

        features = []
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        CLS, SEP = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        EOS, SOS = self.tokenizer.convert_tokens_to_ids(['[EOS]', '[SOS]'])

        chunk_size = int(len(data) / args.chunk_nums)

        for sample in tqdm(data[chunk * chunk_size: (chunk + 1) * chunk_size]):
            if mode == 'test' and sample['label'] <3:
                continue

            content = sample['content']
            abstract = sample['summary']

            # clean content and abstract
            content = clean(content)
            abstract = clean(abstract)

            # tokenize content and title
            content_tokens = self.tokenizer.tokenize(content)
            abstract_tokens = self.tokenizer.tokenize(abstract)
            content_ids = self.tokenizer.convert_tokens_to_ids(content_tokens)
            abstract_ids = self.tokenizer.convert_tokens_to_ids(abstract_tokens)

            # truncate if necessary
            if len(content_ids) > args.content_max_len - 2:  # 2 for [CLS] ... [SEP]
                content_ids = content_ids[:args.content_max_len - 2]

            if len(abstract_ids) > args.summary_max_len - 1: # 1 for [SOS] or [EOS]
                abstract_ids = abstract_ids[:args.summary_max_len - 1]

            input_ids = [CLS] + content_ids + [SEP]
            decoder_input_ids = [SOS] + abstract_ids
            labels = abstract_ids + [EOS]

            features.append({
                'X': input_ids,
                'y_q': decoder_input_ids,
                'mle_labels': labels
            })

        logger.info("Saving examples to cache file {}".format(cache_file))
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)

        return features

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for data.
        :param batch:

            Args:
                batch: batch of data

        :return:  X, y_q, y_k, mle_labels
        """

        batch_size = len(batch)
        max_enc_len = max([len(sample['X']) for sample in batch])
        max_dec_q_len = max([len(sample['y_q']) for sample in batch])

        X = torch.zeros(batch_size, max_enc_len).long()
        y_q = torch.zeros(batch_size, max_dec_q_len).long()
        mle_labels = torch.zeros(batch_size, max_dec_q_len).long()

        for i, sample in enumerate(batch):
            X[i, :len(sample['X'])] = torch.tensor(sample['X'])
            y_q[i, :len(sample['y_q'])] = torch.tensor(sample['y_q'])
            mle_labels[i, :len(sample['mle_labels'])] = torch.tensor(sample['mle_labels'])

        return X, y_q, mle_labels






























