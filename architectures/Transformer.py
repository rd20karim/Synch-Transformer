import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 r,
                 max_length=100,
                 spatial=False):
        super().__init__()

        self.device = device
        self.spatial= spatial
        if self.spatial:
            joint_dim = 3
            n_joint = 22
            s_n_heads =    8  #4
            dim_proj_node = 8  # 32 #16
            s_dm = s_n_heads*dim_proj_node
            s_pf = 2*s_dm   #4*s_dm #2*s_dm
           # Note the Frame wise pose features will have 22*s_dm dimensions
            self.spat_tok_embedding = nn.Linear(joint_dim, s_dm)
            self.spat_pos_embedding = nn.Embedding(n_joint, s_dm)
            self.spat_layers = nn.ModuleList([EncoderLayer(s_dm, s_n_heads,s_pf,dropout,device,r) for _ in range(n_layers)])

            self.tok_embedding = nn.Linear(s_dm*n_joint, hid_dim)
            self.pos_embedding = nn.Embedding(max_length, hid_dim)
            self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device, r) for _ in range(n_layers)])
        else:
            self.tok_embedding = nn.Linear(input_dim, hid_dim)
            self.pos_embedding = nn.Embedding(max_length, hid_dim)
            self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device, r)
                                         for _ in range(n_layers)])


        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(hid_dim)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        self.attention = []
        batch_size = src.shape[0]
        src_len = src.shape[1]

        if self.spatial:
            n_joint = 22
            B, T, CV = src.shape

            src = src.reshape(B * T, n_joint, CV//n_joint)
            pos = torch.arange(0, n_joint).unsqueeze(0).repeat(B*T, 1).to(self.device)
            src = self.dropout((self.spat_tok_embedding(src) * self.scale) + self.spat_pos_embedding(pos))

            for spat_layer,temp_layer in zip(self.spat_layers,self.layers):
              # SPATIAL ENCODING
                # No masking on Node Level
                src,attention = spat_layer(src, None,spatial=self.spatial)
                # src = [batch size*src len,V, hid dim]
                hid_dim = src.shape[-1]
                src = src.reshape(B, T, n_joint * hid_dim)

              # TEMPORAL ENCODING
                pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

                src,attention = temp_layer(src, src_mask)
                self.attention.append(attention)

        else: # No spatial
            pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            # pos = [batch size, src len]
            src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos%100)) # todo reset later
            # src = [batch size, src len, hid dim]
            for layer in self.layers:
                src,attention = layer(src, src_mask)
                self.attention.append(attention)

        self.attention = torch.stack(self.attention,dim=0)
        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 r):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,pf_dim,dropout)
        self.r = r
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask,spatial=False):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # -------------- SPATIAL ENCODING JOINT LEVEL ----------------------
        if spatial:
            # src  :[B*T,V,C_out] Number of nodes become the seq_length

            # ----------- ENCODER TEMPORAL SELF  ATTENTION
            _src, attention = self.self_attention(src, src, src, src_mask,gauss_mask=False,obj=self)
            # dropout, residual connection and layer norm
            src = self.self_attn_layer_norm(src + self.dropout(_src))
            # src = [batch size, src len, hid dim]
            # position-wise feedforward
            _src = self.positionwise_feedforward(src)
            # dropout, residual and layer norm
            src = self.ff_layer_norm(src + self.dropout(_src))

        else:
            # ----------- ENCODER TEMPORAL SELF  ATTENTION
            _src, attention = self.self_attention(src, src, src, src_mask,gauss_mask=True,obj=self)
            # dropout, residual connection and layer norm
            src = self.self_attn_layer_norm(src + self.dropout(_src))
            # src = [batch size, src len, hid dim]
            # position-wise feedforward
            _src = self.positionwise_feedforward(src)
            # dropout, residual and layer norm
            src = self.ff_layer_norm(src + self.dropout(_src))
            # src = [batch size, src len, hid dim]

        return src,attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = np.sqrt(hid_dim)

        self.device = device
    def forward(self, query, key, value, mask=None,
                gauss_mask=False,cross_gauss_mask=False,
                trg_lens=None,mode="superv", obj=None,dilation=1):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # TODO REMOVE THIS FOLLOWING TEMPORARY LINE
        self.values = V # retrieved motion frames representations

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        if gauss_mask:
            energy = self.apply_gauss_mask(energy,r=obj.r)
            # DILATED SELF ATTENTION TAKE EFFECT WHEN DILATION>1
            energy = energy[:,:,::dilation,::dilation]

        # attention = [batch size, n heads, query len, key len]
        if cross_gauss_mask:
            # DILATED CROSS ATTENTION TAKE EFFECT WHEN DILATION>1
            energy = energy[:,:,::dilation,::dilation]
            _attention = torch.softmax(energy, dim=-1)
            B, H, Q_len, K_len = _attention.shape
            frames = torch.arange(K_len,device=self.device,dtype=_attention.dtype)
            # TODO MAKE THIS POSITION SOMEHOW TRAINABLE

            align_pos = torch.round(_attention@frames).unsqueeze(-1)
            #align pos = [batch size, n heads, query len, 1]
            src_lens = mask.float().sum(-1)

            # ------------------ ALIGNMENT ORDER SUPERVISION -----------------------------
            if mode=="superv":

                # [batch size, n heads, query]
                _mean_att = (_attention@frames) # Equivalent to the soft_argmax(energy)
                # difference align position between next and previous words
                margin = obj.m
                order_loss = torch.sum(torch.relu((_mean_att[:,:,:-1]-_mean_att[:,:,1:]+ margin)/src_lens.squeeze(-1))**2,dim=-1)\
                             /(trg_lens-1).unsqueeze(-1)
                # The cross attention will always have one heads for interpretability
                order_loss = order_loss.squeeze(1)  #[bacth_size,]

                #--------------- BATCH LOSS FOR SUPERVISION TO CONSTRAINT STRUCTERED CROSS ATTENTION---------

                # Align first word with the beginning of motion
                self.align0_loss= torch.mean((_mean_att[:,:,0]).squeeze(1))
                # Force a monotonic order of word alignment positions
                self.order_loss = torch.mean(order_loss)
                # Align final word with the end of motion
                self.alignf_loss = torch.mean(   (_mean_att[:,:,-1]/src_lens.squeeze(-1) - 1) **2    )
                #--------------------------------------------------------------------------------------------

            # Align max and min positions, similar to pt-D, pt+D
            lower_bounds = torch.clamp(align_pos-obj.D,min=0)
            upper_bounds = torch.clamp(align_pos+obj.D,max=len(frames)-1)
            upper_mask = torch.arange(K_len,device=self.device).expand(_attention.size())<= upper_bounds
            lower_mask = torch.arange(K_len,device=self.device).expand(_attention.size())>=lower_bounds

            # mask =  [batch size, n heads, query len, key len]
            #cross_mask = (upper_mask & lower_mask).float()

            # Finally apply the Local cross mask as inplace operations
            _energy = energy.clone() # Avoid inplace operation for gradients track
            _energy.masked_fill_(upper_mask.float()==0.,-1e10)
            _energy.masked_fill_(lower_mask.float()==0.,-1e10)

            # New Local cross attention !!
            attention = torch.softmax(_energy, dim=-1)

        else :

            attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention

    def apply_gauss_mask(self,matrix, r):
        # Create a matrix of size n x n filled with zeros
        n= matrix.shape[-1]
        # Create masks for the two conditions
        lower_mask = torch.tril(torch.ones((n, n),device=self.device), diagonal=-r - 1).unsqueeze(0).unsqueeze(0).expand_as(matrix)
        upper_mask = torch.triu(torch.ones((n, n),device=self.device), diagonal=r + 1).unsqueeze(0).unsqueeze(0).expand_as(matrix)
        # Inplace operations
        matrix.masked_fill_(lower_mask == 1, -1e10)
        matrix.masked_fill_(upper_mask == 1, -1e10)

        return matrix


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x



class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 D,
                 margin,
                 device,
                 max_length=100,
                 concat=False):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        Dec_f = DecoderLayer_concat if concat else DecoderLayer
        self.layers = nn.ModuleList([Dec_f(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device,
                                                  D,
                                                  margin)
                                     for _ in range(n_layers)])


        F = 2 if concat else 1
        self.fc_out = nn.Linear(F*hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = np.sqrt(hid_dim) # torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask,mode="train"):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        self.attention = []
        self.cross_attention = []
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        # TODO Infer  self.trg_pad_idx
        trg_lens = (trg != 0).float().sum(-1)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        if mode =="train":
            for layer in self.layers:
                trg, cross_attention,dec_attention = layer(trg, enc_src, trg_mask, src_mask,trg_lens=trg_lens,mode=mode)
                self.cross_attention.append(cross_attention)
                self.attention.append(dec_attention)
                # trg = [batch size, trg len, hid dim]
                # attention = [batch size, n heads, trg len, src len]
                self.order_loss_ = layer.enc_dec_attention.order_loss
                self.align0_loss_ = layer.enc_dec_attention.align0_loss
                self.alignf_loss_ = layer.enc_dec_attention.alignf_loss

            output = self.fc_out(trg)

        ###-----------------------------INFERENCE-----------------------------------###
        else :
            predicted_target = torch.zeros_like(trg)
            predicted_target[:, 0, :] = trg[:, 0, :]
            outputs = []
            for j in range(trg_len):
                # STORE ALL PREVIOUS PREDICTED WORDS
                sos_dec = predicted_target #[:,j,:].unsqueeze(1) #[batch size, 1, hid dim]
                embd_dec = sos_dec

                for layer in self.layers:
                    embd_dec, cross_attention,dec_attention = layer(embd_dec, enc_src, trg_mask, src_mask,trg_lens,mode)
                    self.attention.append(dec_attention)
                    self.cross_attention.append(cross_attention)
                    self.order_loss_ = 0
                    self.align0_loss_ = 0

                logits = self.fc_out(embd_dec[:,j,:].unsqueeze(1))
                outputs.append(logits)
                word_ids = torch.argmax(logits,dim=2)
                # NEXT GENERATED WORDS
                if j<trg_len-1:
                    # STORE ALL PREVIOUS PREDICTED WORDS
                    predicted_target[:,j+1,:] = self.dropout((self.tok_embedding(word_ids.squeeze(1)) * self.scale)
                                                             + self.pos_embedding(pos[:,j+1]))
                #CONCATENATE ALL THE PREDICTED WORDS
            output = torch.cat(outputs,dim=1)

        self.attention = torch.stack(self.attention,dim=0)
        self.cross_attention = torch.stack(self.cross_attention,dim=0)

        # output = [batch size, trg len, output dim]

        return output, cross_attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 D,
                 margin):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        self.enc_dec_attention = MultiHeadAttentionLayer(hid_dim, 1, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.D = D
        self.m = margin
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask,trg_lens=None,mode="train"):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # ------------ SELF DECODER ATTENTION
        _trg, dec_attention = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # ------------ CROSS ATTENTION
        _trg, cross_att = self.enc_dec_attention(trg, enc_src, enc_src, src_mask,
                                                 cross_gauss_mask=True,
                                                 trg_lens=trg_lens,
                                                 mode="superv" if mode == "train" else "nosuperv",
                                                 obj=self)

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        return trg, cross_att,dec_attention

class DecoderLayer_concat(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 D,
                 margin):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        self.motion_layer_norm = nn.LayerNorm(hid_dim)
        self.language_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.enc_dec_attention = MultiHeadAttentionLayer(hid_dim, 1, dropout, device)

        self.ff_layer_norm = nn.LayerNorm(hid_dim+hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim+hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.D = D
        self.m = margin
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask,trg_lens=None,mode="train"):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # ------------ SELF DECODER ATTENTION
        _trg, dec_attention = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # ------------ CROSS ATTENTION
        _trg, cross_att = self.enc_dec_attention(trg, enc_src, enc_src, src_mask,
                                                 cross_gauss_mask=True,
                                                 trg_lens=trg_lens,
                                                 mode="superv" if mode=="train" else "nosuperv",
                                                 obj=self)

        # Concatenate motion and language information
        trg = torch.cat([self.motion_layer_norm(trg),self.language_layer_norm(self.dropout(_trg))],dim=-1)
        # trg = [batch size, trg len, 2* hid dim]
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]


        return trg, cross_att,dec_attention

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device,
                 beam_search=False,
                 n_joint=22):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.beam_search = beam_search
        self.n_joint = n_joint

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx)[:, :, 0].unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg, src_len=None,mode="train"):

        # src = [batch size, src len]
        # trg = [batch size, trg len]
        if src_len is not None:
            src_mask = (src != torch.tensor([0] * self.n_joint*3, device=self.device))[:, :, 0].unsqueeze(1).unsqueeze(2)
        else:
            src_mask = self.make_src_mask(src)

        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, cross_attention = self.decoder(trg, enc_src, trg_mask, src_mask,mode=mode)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, cross_attention


def create_model(OUTPUT_DIM,DEC_LAYERS,DEC_HEADS,DEC_PF_DIM,DEC_DROPOUT,
                  INPUT_DIM,HID_DIM,ENC_LAYERS,ENC_HEADS,ENC_PF_DIM,ENC_DROPOUT,
                  device, max_length=1400,D=10,r=2,margin=1,n_joint=22,spatial=False,concat=False):

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  r,
                  max_length=max_length,
                  spatial=spatial)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  D,
                  margin,
                  device,
                  concat=concat)


    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device,n_joint=n_joint).to(device)

    return model
