import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems, reset_state.xy_img)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None


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

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        fusion_layer_num = self.model_params['fusion_layer_num']

        self.embedding = nn.Linear(6, embedding_dim)
        self.embedding_patch = PatchEmbedding(**model_params)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
        self.layers_img = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num - fusion_layer_num)])
        self.fusion_layers = nn.ModuleList([EncoderFusionLayer(**model_params) for _ in range(fusion_layer_num)])
        self.fcp = nn.Parameter(torch.randn(1, self.model_params['bn_num'], embedding_dim))
        self.fcp_img = nn.Parameter(torch.randn(1, self.model_params['bn_img_num'], embedding_dim))

    def forward(self, data, img):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        embedded_patch = self.embedding_patch(img)

        out = embedded_input
        out_img = embedded_patch
        for i in range(self.model_params['encoder_layer_num'] - self.model_params['fusion_layer_num']):
            out = self.layers[i](out)
            out_img = self.layers_img[i](out_img)
        fcp = self.fcp.repeat(data.shape[0], 1, 1)
        fcp_img = self.fcp_img.repeat(img.shape[0], 1, 1)
        for layer in self.fusion_layers:
            out, out_img, fcp, fcp_img = layer(out, out_img, fcp, fcp_img)
        return torch.cat((out, out_img), dim=1)

class PatchEmbedding(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.img_size = self.model_params['img_size']
        self.patch_size = self.model_params['patch_size']
        self.in_channels = self.model_params['in_channels']
        self.embed_dim = self.model_params['embedding_dim']

        # patch num
        self.patches = self.img_size // self.patch_size

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

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

class EncoderFusionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.Wq_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_img = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine_img = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1_img = Add_And_Normalization_Module(**model_params)
        self.feedForward_img = Feed_Forward_Module(**model_params)
        self.addAndNormalization2_img = Add_And_Normalization_Module(**model_params)

    def forward(self, input, input_img, fcp, fcp_img):
        input1 = torch.cat((input, fcp), dim=1)
        input1_img = torch.cat((input_img, fcp_img), dim=1)

        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        v = reshape_by_heads(self.Wv(torch.cat((input1, fcp_img), dim=1)), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        q_img = reshape_by_heads(self.Wq_img(input1_img), head_num=head_num)
        k_img = reshape_by_heads(self.Wk_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        v_img = reshape_by_heads(self.Wv_img(torch.cat((input1_img, fcp), dim=1)), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat_img = multi_head_attention(q_img, k_img, v_img)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out_img = self.multi_head_combine_img(out_concat_img)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1_img = self.addAndNormalization1_img(input1_img, multi_head_out_img)
        out2_img = self.feedForward_img(out1_img)
        out3_img = self.addAndNormalization2_img(out1_img, out2_img)

        return out3[:, :-self.model_params['bn_num']], out3_img[:, :-self.model_params['bn_img_num']], out3[:, -self.model_params['bn_num']:], out3_img[:, -self.model_params['bn_img_num']:]        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        hyper_input_dim = 3
        hyper_hidden_embd_dim = self.model_params['hyper_hidden_dim']
        self.embd_dim = 3
        self.hyper_output_dim = 5 * self.embd_dim
        
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)
        
        self.hyper_Wq_first = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * embedding_dim, bias=False)

        self.Wq_last_para = None
        self.multi_head_combine_para = None

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        
    def assign(self, pref):
        
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        
        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)
        
        
    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        num_patches = (self.model_params['img_size'] // self.model_params['patch_size']) * (
                    self.model_params['img_size'] // self.model_params['patch_size'])
        node_size = encoded_nodes.shape[1] - num_patches

        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=head_num)

        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes[:, :node_size].transpose(1, 2)
        # shape: (batch, embedding, problem)
     
    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(F.linear(encoded_last_node, self.Wq_last_para), head_num = head_num)
       
        q = self.q_first + q_last 
        # shape: (batch, head_num, pomo, qkv_dim)

        num_patches = (self.model_params['img_size'] // self.model_params['patch_size']) * (
                self.model_params['img_size'] // self.model_params['patch_size'])
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=torch.cat(
            (ninf_mask, torch.zeros(ninf_mask.shape[0], ninf_mask.shape[1], num_patches)), dim=-1))
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)
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


class Add_And_Normalization_Module(nn.Module):
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


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
