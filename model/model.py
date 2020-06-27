import itertools
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.net_vlad import NetVLAD
import ipdb
from torch import autograd

class MoEE(BaseModel):
    def __init__(self, label, experts_used, expert_dims, aggregation_method, projection_dim, pretrained, use_moe):
        super().__init__()
        self.n_clips = 1
        self.label = label
        self.experts_used = experts_used.copy()
        self.experts_used.remove(self.label)
        expert_dims['label'] = expert_dims[label]
        self.expert_dims = expert_dims
        self.aggregation_method = aggregation_method
        self.projection_dim = projection_dim

        self.aggregation = nn.ModuleDict({
            expert: get_aggregation(info, self.expert_dims[expert]) for expert, info in self.aggregation_method.items()
            if expert in self.experts_used + ['label']
        })

        for key in self.aggregation_method:
            if self.aggregation_method[key]['type'] == "net_vlad":
                self.expert_dims[key] = self.expert_dims[key] * self.aggregation_method[key][
                    'cluster_size']  # TODO: hacky, improve

        self.video_GU = nn.ModuleDict({
            expert: Gated_Embedding_Unit(self.expert_dims[expert], self.projection_dim, channels=self.n_clips)
            for expert in self.experts_used
        })

        self.clip_GU = nn.ModuleList([
            nn.Identity() for clip in range(self.n_clips)
        ])

        self.text_GU = nn.ModuleDict({
            expert: Gated_Embedding_Unit(self.aggregation['label'].out_dim, self.projection_dim, channels=0)
            for expert in experts_used

        })

        self.text_clip = nn.ModuleList([
            nn.Identity() for clip in range(self.n_clips)
        ])

        self.moe_fc = nn.Linear(self.expert_dims['label'], len(self.experts_used)* self.n_clips)

    def get_moe_scores(self, text):
        res = F.softmax(self.moe_fc(text), dim=-1)
        return res

    def forward(self, x, evaluation=False, debug=False):
        '''
        :param x: Dictionary of experts and one of the experts 'label'.
                Each expert has keys 'ftr', 'missing' and 'n_tokens'
                x[expert]['ftr']: "b x clips x ftr" OR
                                    "b x clips x n_tokens x ftr"

                x[expert]['missing']: b x clips   boolean tensor, True if expert is missing for that clip
                x[expert]['n_tokens']: number of actual tokens for that expert in that clip (we need this because of padding)

        :return: Similarity score for batch: b x b (return other stuff if evaluating: moe weights, text embed, video embed
        '''

        missing = []

        res = {}
        video_experts = []
        for expert in x:
            ftr = x[expert]['ftr']
            miss = x[expert]['missing']
            if expert == 'label':
                ftr = ftr.squeeze(1)
                ftr = self.aggregation[expert](ftr, x[expert]['n_tokens'])
                # ftr = ftr.mean(dim=1)
                text = ftr
            else:
                if len(ftr.shape) == 4:
                    n_tokens = x[expert]['n_tokens']
                    ftr = self.aggregation[expert](ftr, n_tokens)
                res[expert] = ftr
                missing.append(miss)
                video_experts.append(expert)

        missing = torch.stack(missing, dim=1).bool()  # b, expert, clip
        text_embed = []
        video_embed = []
        for idx, expert in enumerate(video_experts):
            video_embed.append(self.video_GU[expert](res[expert]))
            text_embed.append(self.text_GU[expert](text))

        video_embed = torch.stack(video_embed, dim=2)  # b, n_clips, experts, ftr_dim
        text_embed = torch.stack(text_embed, dim=1)  # b, expert, ftr_dim


        batch_sz = video_embed.shape[0]

        video_embed_mod = []
        text_embed_mod = []
        for idx in range(self.n_clips):
            video_embed_mod.append(self.clip_GU[idx](video_embed[:, idx]))  # clip-level GU
            text_embed_mod.append(self.text_clip[idx](text_embed))

        video_embed_mod = torch.stack(video_embed_mod, dim=2)  # b, expert, clip, ftr_dim
        text_embed_mod = torch.stack(text_embed_mod, dim=2)

        video_embed_mod = F.normalize(video_embed_mod, dim=-1)
        text_embed_mod = F.normalize(text_embed_mod, dim=-1)

        moe_weights = self.get_moe_scores(text)
        moe_weights = moe_weights.view(-1, len(self.experts_used), self.n_clips)  # b, expert, clip
        moe_weights = moe_weights.unsqueeze(1).repeat(1, batch_sz, 1, 1)  # text, video, expert, clip

        missing = missing.unsqueeze(0)  # 1, video, expert, clip

        missing = missing.repeat(batch_sz, 1, 1, 1)

        moe_weights = moe_weights.masked_fill(missing, 0)
        norm_weights = torch.sum(moe_weights, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        moe_weights = torch.div(moe_weights, norm_weights)

        embed_stack = torch.einsum('tecd,vecd->tvec', [text_embed_mod, video_embed_mod])  # text x video x expert x clip
        embed_stack = embed_stack * moe_weights  # tvec similarity scores
        conf_mat = embed_stack.sum(dim=(2, 3))  # sum over e,c

        if evaluation:
            return conf_mat, video_embed_mod, text_embed_mod, moe_weights

        return conf_mat

class Collaborative_Gating_Unit(nn.Module):
    def __init__(self, output_dimension, num_inputs, number_g_layers, number_h_layers, use_bn_reason):
        super(Collaborative_Gating_Unit, self).__init__()
        self.num_g_layers = number_g_layers
        self.num_h_layers = number_h_layers
        self.use_bn_reason = use_bn_reason
        self.output_dim = output_dimension

        self.g_reason_shared = self.instantiate_reason_module()
        self.g_reason_1 = nn.Linear(output_dimension * num_inputs, output_dimension)
        self.h_reason_shared = self.instantiate_reason_module()

    def instantiate_reason_module(self):
        g_reason_shared = []
        for _ in range(self.num_g_layers - 1):
            if self.use_bn_reason:
                g_reason_shared.append(nn.BatchNorm1d(self.output_dim))
            g_reason_shared.append(nn.ReLU())
            g_reason_shared.append(nn.Linear(self.output_dim, self.output_dim))

        return nn.Sequential(*g_reason_shared)

    def common_project(self, x):
        return self.fc(x)

    def forward(self, cp1, cp2):
        pass



class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True, channels=0):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension, channels)
        self.gating = gating

    def forward(self, x):
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        x = F.normalize(x, dim=-1)

        return x

class Gated_Embedding_Unit_Reasoning(nn.Module):
    def __init__(self, output_dimension, n_clips):
        super(Gated_Embedding_Unit_Reasoning, self).__init__()
        self.cg = ContextGatingReasoning(output_dimension, n_clips)

    def forward(self, x, mask):
        x = self.cg(x, mask)
        x = F.normalize(x)
        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, channels, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.channels = channels
        if channels > 0:
            bn_dim = channels
        else:
            bn_dim = dimension
        self.batch_norm = nn.BatchNorm1d(bn_dim)

    def forward(self, x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1)

        x = torch.cat((x, x1), -1)

        return F.glu(x, -1)

class ContextGatingReasoning(nn.Module):
    def __init__(self, dimension, n_clips, add_batch_norm=True):
        super(ContextGatingReasoning, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(n_clips)
        self.batch_norm2 = nn.BatchNorm1d(n_clips)
    def forward(self, x, x1):

        x2 = self.fc(x)

        # t = x1 + x2
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
            x2 = self.batch_norm2(x2)
            # t = self.batch_norm (t)

        # t = (F.sigmoid(x1) + F.sigmoid (x2))/2

        t = x1 + x2

        # t = (t > 0.2).float() * 1
        # t = th.trunc(2*F.sigmoid (t)-0.5)
        # print (t)
        # return x*F.sigmoid(t)

        # return t  (curr no sigmoid hoho!)
        x = torch.cat((x, t), -1)
        return F.glu(x, -1)

class ReduceDim(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

#         self.fc = nn.Linear(input_dimension, 512)
#         self.fc2 = nn.Linear(512, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        #         x = self.fc2(F.relu(x))
        x = F.normalize(x)
        return x


class Debug(BaseModel):
    def __init__(self, experts_used):
        super().__init__()
        self.weight_dict = nn.ModuleDict({
            key: nn.Conv2d(1, 1, 1, stride=1, bias=False) for key in experts_used
        })
        self.ftrs = list(experts_used)

    def forward(self, x, target):
        return cosine_similarity(x['clip_name']['ftrs'], target)


class ScalarWeight(nn.Module):
    def __init__(self, init_val=1, len=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(len) * init_val)

    def forward(self, x):
        return x * self.weight.unsqueeze(-1)


class MeanToken(nn.Module):
    def __init__(self, dim_idx):
        super().__init__()
        self.dim_idx = dim_idx

    def forward(self, x, n_tokens):
        n_dims = len(x.shape)
        if n_dims == 3:
            x_sum = x.sum(dim=1)
            x_mean = x_sum * n_tokens.unsqueeze(1).float().pow(-1)
        elif n_dims == 4:
            x_sum = x.sum(dim=2)
            x_mean = x_sum * n_tokens.unsqueeze(2).float().pow(-1)
        else:
            x_sum = x.sum(dim=2)
            x_mean = x_sum * n_tokens.unsqueeze(2).float().pow(-1)

        return x_mean


class MaxToken(nn.Module):
    def __init__(self, dim_idx):
        super().__init__()
        self.dim_idx = dim_idx

    def forward(self, x, n_tokens):
        return x.max(dim=self.dim_idx)


def get_aggregation(agg, feature_size):
    if agg['type'] == 'net_vlad':
        cluster_size = agg['cluster_size']
        ghost_clusters = agg['ghost_clusters']
        return NetVLAD(cluster_size, feature_size, ghost_clusters)
    elif agg['type'] == 'mean':
        return MeanToken(1)
    elif agg['type'] == 'max':
        return MaxToken(1)
    else:
        raise NotImplementedError


def cosine_similarity(content, text, eps=1e-8):
    b1, d1 = content.shape
    b2, d2 = text.shape
    assert (b1 == b2 and d1 == d2)

    # TODO: Use lens instead of zero checking.

    content_norm = torch.norm(content, dim=1).pow(-1)  # b x n_c TODO: For end-to-end, make this 0-div safe. Add epsilon
    text_norm = torch.norm(text, dim=1).pow(-1)  # b x n_s

    # content_norm = content_norm.masked_fill_(content_padding, 1)
    # text_norm = text_norm.masked_fill_(text_padding, 1)
    dot_prod = torch.einsum('ad,bd->ab', content, text)  # similarity between every sample in batch
    cosine_sim = dot_prod * (content_norm * text_norm + torch.ones_like(text_norm) * eps)

    return cosine_sim


def sim_matrix(a, b, weights=None, eps=1e-8):
    """
    added eps for numerical stability
    """
    if len(a.shape) == 3 and len(b.shape) == 3:
        # MoEE
        embed_stack = torch.einsum('ted,ved->tve', [a, b])
        if weights is not None:
            embed_stack = embed_stack * weights
        return embed_stack.sum(dim=1)

    elif len(a.shape) == 4 and len(b.shape) == 4:
        # a (query): text x expert x proj_dim
        # b (video): video x expert x clips x proj_dim
        # weights (moee): text x video x experts x clips
        # MoEE^2
        embed_stack = torch.einsum('tecd,vecd->tvec', [a, b])  # text x video x expert x clip

        # embed_stack = torch.transpose(embed_stack, 1, 2)
        if weights is not None:
            embed_stack = embed_stack * weights
        return embed_stack.sum(dim=(2, 3))

    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    if torch.isnan(sim_mt).any():
        print('nans found in similarity matrix')
    return sim_mt


if __name__ == '__main__':
    random_tensor = torch.rand((32, 3, 4, 512))

    print('Custom distance function works as expected')
