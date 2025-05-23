import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes_kv = None
        self.encoded_nodes_q = None
        # shape: (batch, problem+1, EMBEDDING_DIM)
        embedding_dim = self.model_params['embedding_dim']
        hyper_hidden_embd_dim = 256

        self.hyper_fc2 = nn.Linear(embedding_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, embedding_dim, bias=True)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        pref = reset_state.preference
        img = reset_state.depot_node_demand_img

        self.encoded_nodes_q = self.encoder(depot_xy, node_xy_demand, pref, img)
        batch_size, problem_size, _ = node_xy.size()
        embedding_dim = self.model_params['embedding_dim']

        # hyper_embd = self.hyper_fc1(pref)
        encoded_ps = position_encoding_init(batch_size, problem_size, embedding_dim, pref.device)
        EP_embedding = self.hyper_fc2(encoded_ps)
        EP_embed = self.hyper_fc3(EP_embedding)
        self.encoded_nodes_kv = self.encoded_nodes_q
        self.encoded_nodes_kv[:, :problem_size] = self.encoded_nodes_q[:, :problem_size] + EP_embed
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes_kv)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes_q, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        fusion_layer_num = self.model_params['fusion_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.embedding_pref = nn.Linear(2, embedding_dim)
        self.embedding_patch = PatchEmbedding(**model_params)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
        self.layers_img = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
        self.fusion_layers = nn.ModuleList([EncoderFusionLayer(**model_params) for _ in range(fusion_layer_num)])
        self.fcp = nn.Parameter(torch.randn(1, self.model_params['bn_num'], embedding_dim))
        self.fcp_img = nn.Parameter(torch.randn(1, self.model_params['bn_img_num'], embedding_dim))

    def forward(self, depot_xy, node_xy_demand, pref, img):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)
        embedded_pref = self.embedding_pref(pref)

        embedded_patch = self.embedding_patch(img)

        out = torch.cat((embedded_depot, embedded_node, embedded_pref[:, None, :]), dim=1)
        out_img = torch.cat((embedded_patch, embedded_pref[:, None, :]), -2)
        for i in range(self.model_params['encoder_layer_num'] - self.model_params['fusion_layer_num']):
            out = self.layers[i](out)
            out_img = self.layers_img[i](out_img)
        fcp = self.fcp.repeat(depot_xy.shape[0], 1, 1)
        fcp_img = self.fcp_img.repeat(img.shape[0], 1, 1)
        for layer in self.fusion_layers:
            out, out_img, fcp, fcp_img = layer(out, out_img, fcp, fcp_img)
        return torch.cat((out[:, :-1], out_img[:, :-1]), dim=1)

class PatchEmbedding(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.patch_size = self.model_params['patch_size']
        self.in_channels = self.model_params['in_channels']
        self.embed_dim = self.model_params['embedding_dim']


        self.proj = nn.Linear(self.patch_size * self.patch_size * self.in_channels, self.embed_dim)

        # positional embedding
        self.position_proj = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.in_channels, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * self.in_channels)

        # patch embedding
        embedded_patches = self.proj(patches)

        # add positional embedding
        grid_x, grid_y = torch.meshgrid(torch.arange(self.patches), torch.arange(self.patches), indexing='ij')  # 'ij' indexing for row-major order
        xy = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        xy = xy / (self.patches - 1)
        embedded_patches += self.position_proj(xy)

        return embedded_patches

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        embed_nodes = input1[:, :-1, :]  # (batch, problem, embedding_dim)
        pref_node = input1[:, -1, :][:, None, :]  # (batch, 1, embedding_dim)

        q1 = reshape_by_heads(self.Wq1(input1), head_num=head_num)
        k1 = reshape_by_heads(self.Wk1(input1), head_num=head_num)
        v1 = reshape_by_heads(self.Wv1(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        q2 = reshape_by_heads(self.Wq2(embed_nodes), head_num=head_num)
        k2 = reshape_by_heads(self.Wk2(pref_node), head_num=head_num)
        v2 = reshape_by_heads(self.Wv2(pref_node), head_num=head_num)

        out_concat = multi_head_attention(q1, k1, v1)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        add_concat = multi_head_attention(q2, k2, v2)
        out_concat[:, :-1] = out_concat[:, :-1] + add_concat

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)

class EncoderFusionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = AddAndInstanceNormalization(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = AddAndInstanceNormalization(**model_params)

        self.Wq1_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk1_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv1_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq2_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk2_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv2_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine_img = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1_img = AddAndInstanceNormalization(**model_params)
        self.feedForward_img = FeedForward(**model_params)
        self.addAndNormalization2_img = AddAndInstanceNormalization(**model_params)

    def forward(self, input, input_img, fcp, fcp_img):
        input1 = torch.cat((fcp, input), dim=1)
        input1_img = torch.cat((fcp_img, input_img), dim=1)
        head_num = self.model_params['head_num']
        embed_nodes = input[:, :-1, :]  # (batch, problem, embedding_dim)
        pref_node = input[:, -1, :][:, None, :]  # (batch, 1, embedding_dim)

        q1 = reshape_by_heads(self.Wq1(input1), head_num=head_num)
        k1 = reshape_by_heads(self.Wk1(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        v1 = reshape_by_heads(self.Wv1(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        q2 = reshape_by_heads(self.Wq2(embed_nodes), head_num=head_num)
        k2 = reshape_by_heads(self.Wk2(pref_node), head_num=head_num)
        v2 = reshape_by_heads(self.Wv2(pref_node), head_num=head_num)

        out_concat = multi_head_attention(q1, k1, v1)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        add_concat = multi_head_attention(q2, k2, v2)
        out_concat[:, self.model_params['bn_num']:-1] = out_concat[:, self.model_params['bn_num']:-1] + add_concat

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        embed_nodes_img = input_img[:, :-1, :]  # (batch, problem, embedding_dim)
        pref_node_img = input_img[:, -1, :][:, None, :]  # (batch, 1, embedding_dim)

        q1_img = reshape_by_heads(self.Wq1_img(input1_img), head_num=head_num)
        k1_img = reshape_by_heads(self.Wk1_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        v1_img = reshape_by_heads(self.Wv1_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        q2_img = reshape_by_heads(self.Wq2_img(embed_nodes_img), head_num=head_num)
        k2_img = reshape_by_heads(self.Wk2_img(pref_node_img), head_num=head_num)
        v2_img = reshape_by_heads(self.Wv2_img(pref_node_img), head_num=head_num)

        out_concat_img = multi_head_attention(q1_img, k1_img, v1_img)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        add_concat_img = multi_head_attention(q2_img, k2_img, v2_img)
        out_concat_img[:, self.model_params['bn_img_num']:-1] = out_concat_img[:, self.model_params['bn_img_num']:-1] + add_concat_img

        multi_head_out_img = self.multi_head_combine_img(out_concat_img)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1_img = self.addAndNormalization1_img(input1_img, multi_head_out_img)
        out2_img = self.feedForward_img(out1_img)
        out3_img = self.addAndNormalization2_img(out1_img, out2_img)


        return out3[:, self.model_params['bn_num']:], out3_img[:, self.model_params['bn_img_num']:], out3[:, :self.model_params['bn_num']], out3_img[:, :self.model_params['bn_img_num']]
        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.hyper_Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        node_size = encoded_nodes.shape[1] - self.patches ** 2

        self.k = reshape_by_heads(self.hyper_Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.hyper_Wv(encoded_nodes), head_num=head_num)

        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes[:, :node_size].transpose(1, 2)
        # shape: (batch, embedding, problem)

    def forward(self, encoded_last_node, load, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.hyper_Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=torch.cat(
            (ninf_mask, torch.zeros(ninf_mask.shape[0], ninf_mask.shape[1], self.patches ** 2)), dim=-1))
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.hyper_multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.gelu(self.W1(input1)))

def position_encoding_init(batch_szie, n_position, emb_dim, device):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = torch.FloatTensor(np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(200)])).to(device)

    position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    position_encoding = position_enc[n_position - 1]
    return position_encoding[None, None, :].expand(batch_szie, 1, emb_dim)