""" MoviePlots dataset module.

This code is loosely based from the collaborative-experts dataloaders:
https://github.com/albanie/collaborative-experts/tree/master/data_loader

"""
import os
from os.path import join as osj
import ast
import ipdb
import itertools

import pandas as pd
import numpy as np
import torch
import nltk
import pdb
from torch.utils.data import Dataset
from utils.util import memcache, memory_summary


class MovieClips(Dataset):

    def __init__(self, data_dir, metadata_dir, label, experts_used, experts, max_tokens, split='train'):
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.experts_used = [expert for expert in experts_used if experts_used[expert]]
        self.label = label
        if self.label not in experts_used:
            raise ValueError('Label expert must be used.')
        self.experts = experts
        self.expert_dims = self._expert_dims()
        self.max_tokens = max_tokens
        self.split = split
        self._load_metadata()
        self._load_data()

    def _load_metadata(self):
        data = {
            'movies': pd.read_csv(osj(self.metadata_dir, 'movies.csv')).set_index('imdbid'),
            'casts': pd.read_csv(osj(self.metadata_dir, 'casts.csv')).set_index('imdbid'),
            'clips': pd.read_csv(osj(self.metadata_dir, 'clips.csv')).set_index('videoid'),
            'descs': pd.read_csv(osj(self.metadata_dir, 'descriptions.csv')).set_index('videoid'),

        }
        # filter by split {'train', 'val', 'test'}
        split_data = pd.read_csv(osj(self.metadata_dir, 'split.csv')).set_index('imdbid')
        if self.split == 'train_val':
            ids = split_data[split_data['split'].isin(['train', 'val'])].index
        else:
            ids = split_data[split_data['split'] == self.split].index
        for key in data:
            if 'imdbid' in data[key]:
                filter = data[key]['imdbid'].isin(ids)
            else:
                filter = data[key].index.isin(ids)
            data[key] = data[key][filter]

        # Remove inappropriate data
        #empty_clips = pd.read_csv(osj(self.metadata_dir, 'empty_vids.csv')).set_index('videoid')
        #data['clips'] = data['clips'][~data['clips'].index.isin(empty_clips.index)]
        # duplicated descriptions are probably errors by the channel
        data['descs'].dropna(subset=['description'], inplace=True)
        data['descs'].drop_duplicates(subset=['description'], keep=False, inplace=True)

        # remove clips without descriptions (since this is supervised)...
        if self.label == 'description':
            data['clips'] = data['clips'][data['clips'].index.isin(data['descs'].index)]
        elif self.label == 'plot':
            data['clips'] = data['clips'][data['clips']['imdbid'].isin(data['plots'].index)]
        else:
            raise NotImplementedError('Change data removal technique to remove clips without...')

        self.data = data

    def _load_data(self):
        self.expert_data = {}
        for expert in self.experts_used:
            if expert != 'context':
                data_pth = osj(self.data_dir, 'features', self.experts[expert])
                self.expert_data[expert] = memcache(data_pth)
                memory_summary()

        clips_with_data = []
        for expert in self.expert_data:
            if expert != 'description' and expert != 'label':
                clips_with_data += self.expert_data[expert].keys()


        # debugging (input random tensors)
        random = False
        if random:
            for expert in self.expert_data:
                for videoid in self.expert_data[expert]:
                    self.expert_data[expert][videoid] = np.random.randn(*self.expert_data[expert][videoid].shape)

        # debugging (input zero tensors)
        zeros = False
        if zeros:
            for expert in self.expert_data:
                for videoid in self.expert_data[expert]:
                    self.expert_data[expert][videoid] = np.zeros(self.expert_data[expert][videoid].shape)


        clips_with_data = set(clips_with_data)

        #sanity check
        #pdb.set_trace()
        #if not self.data['clips'].index.isin(clips_with_data).all():
        #    print(self.data['clips'][~self.data['clips'].index.isin(clips_with_data)].index)
        #    raise NotImplementedError
        self.data['clips'] = self.data['clips'][self.data['clips'].index.isin(clips_with_data)]
        print(f'{self.split} size: {len(self.data["clips"])} clips')

    def __len__(self):
        return len(self.data['clips'])

    def __getitem__(self, item):
        videoid = self.data['clips'].iloc[item].name

        data = {}
        for expert in self.experts_used:
            packet = self._get_expert_ftr(expert, videoid)
            if expert == self.label:
                data['label'] = packet
            else:
                data[expert] = packet

        id = {'imdbid': self.data['clips'].loc[videoid]['imdbid'], 'videoid': videoid}
        return data, id

    def _get_expert_ftr(self, expert, videoid, context=False):
        packet = {}

        if expert == 'plot':
            videoid = self.data['clips'].loc[videoid]['imdbid']  # TODO: maybe this breaks for clips with no imdbid?

        if videoid not in self.expert_data[expert]:
            missing = True
            some_entry = list(self.expert_data[expert].keys())[0]
            ftr = np.zeros_like(self.expert_data[expert][some_entry])
        else:
            missing = False
            ftr = self.expert_data[expert][videoid]
            #if context:
            #    ftr = np.zeros(ftr.shape)
                #ftr = np.random.randn(*ftr.shape)


        ftr = torch.from_numpy(ftr)
        ftr = ftr.float()
        if len(ftr.shape) == 1:
            pass
        elif len(ftr.shape) == 2:
            ftr, n_tokens = self._pad_to_max_tokens(ftr, expert)
            packet['n_tokens'] = torch.Tensor([n_tokens])
        else:
            raise ValueError

        packet['ftr'] = ftr.unsqueeze(dim=0)
        packet['missing'] = torch.Tensor([missing])
        return packet

    def _pad_to_max_tokens(self, array, expert):
        n_tokens, dim = array.shape
        if n_tokens >= self.max_tokens[expert]:
            res = array[:self.max_tokens[expert]]
            n_tokens = self.max_tokens[expert]
        else:
            res = torch.zeros((self.max_tokens[expert], dim))
            res[:n_tokens] = array
        return res, n_tokens

    def _characters_txt(self, texts, clean_cast):
        raise NotImplementedError

    def _clean_cast(self, cast):
        for actor in cast:
            char = cast[actor]
            char = char.replace('(voice)', '')
            char = char.strip()
            char = [c.strip() for c in char.split('/')]  # deals with one-many actor
            cast[actor] = char  # char here is a list
        return cast

    def _expert_dims(self):
        expert_dims = {
            'BERT': 1024,
            'I3D': 1024,
            'DenseNet-161': 2208,
            'SE-ResNet-154': 2048,
            'S3DG': 1024,
            'SE-ResNet-50': 256,
            '': None
        }
        ftrs_dim = {}
        for key in self.experts:
            arch = self.experts[key].split('/')[0]
            if arch not in expert_dims and arch != "":
                raise ValueError('Expert not found in dims dict, please update')
            ftrs_dim[key] = expert_dims[arch]

        return ftrs_dim
