#-*-coding:utf-8-*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class AMER(BaseModel):
    def __init__(self, config):
        super().__init__()

    def initialize(self, config, device):

        n_classes = 9 if config['emo_type'] == 'primary' else 14
        D_e = config["model"]["args"]["D_e"] #128
        D_v = config["visual"]["dim_env"] + config["visual"]["dim_face"] + config["visual"]["dim_obj"] #4302
        D_a = config["audio"]["feature_dim"] #6373
        D_t = config["text"]["feature_dim"]  #1024
        D_p = config["personality"]["feature_dim"] #118
        
        self.attn = ScaledDotProductAttention((4 * D_e) ** 0.5, attn_dropout=0)

        self.enc_v = nn.Sequential(
            nn.Linear(D_v, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, D_e * 3),
            nn.ReLU(),
            nn.Linear(D_e * 3, 2 * D_e),
        )

        self.enc_a = nn.Sequential(
            nn.Linear(D_a, D_e * 8),
            nn.ReLU(),
            nn.Linear(D_e * 8, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )
        self.enc_t = nn.Sequential(
            nn.Linear(D_t, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.enc_p = nn.Sequential(
            nn.Linear(D_p, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(4 * D_e, 2 * D_e), 
            nn.ReLU(), 
            nn.Linear(2 * D_e, n_classes)
        )

        unified_d = 14 * D_e

        self.fusion_layer = nn.Linear(unified_d, 4 * D_e)

    def forward(self, U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c): #M_v: mask_v
        # Encoders
        V_e, A_e, T_e, P_e = self.enc_v(U_v), self.enc_a(U_a), self.enc_t(U_t), self.enc_p(U_p) #256

        U_all = []
        print('m_v shape...',M_v.shape) #(8,15)
        print('m_a shape...', M_a.shape) #(8,15)
        print('m_t shape...', M_t.shape) #(8,15)
        print('target_loc shape...', target_loc.shape) #(8,15)
        for i in range(M_v.shape[0]):
            target_moment, target_character = -1, -1
            for j in range(target_loc.shape[1]):
                if target_loc[i][j] == 1:
                    target_moment = j % int(seg_len[i].cpu().numpy())
                    target_character = int(j / seg_len[i].cpu().numpy())
                    print('target_moment...', target_moment) #2
                    print('target_character...', target_character) #1
                    break
            
            print('seq_len shape....', seq_lengths.shape)
            print('n_c shape....', n_c.shape)
            inp_V = V_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_T = T_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_A = A_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_P = P_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            print('inp_V shape...',inp_V.shape) #（5，3，256）
            print('inp_T shape...',inp_T.shape) #（5，3，256）
            print('inp_A shape...', inp_A.shape) #（5，3，256）
            print('inp_P shape...', inp_P.shape) #（5，3，256）

            mask_V = M_v[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_T = M_t[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_A = M_a[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            print('mask_V shape...', mask_V.shape) #(5,3)
            print('mask_T shape...',mask_T.shape) #(5,3)
            print('mask_A shape...',mask_A.shape) #(5,3)

            # Concat with personality embedding
            inp_V = torch.cat([inp_V, inp_P], dim=2) #将V，A，T与P组合，然后下一步进行 modality-level attention
            inp_A = torch.cat([inp_A, inp_P], dim=2)
            inp_T = torch.cat([inp_T, inp_P], dim=2)

            U = []

            print('n_c shape....', n_c.shape)
            for k in range(n_c[i]): # 对于每一个character
                new_inp_A, new_inp_T, new_inp_V = inp_A.clone(), inp_T.clone(), inp_V.clone(),
                
                # Modality-level inter-personal attention
                print('seg_len shape...', seg_len.shape)
                for j in range(seg_len[i]): #对于每一个moment
                    att_V, _ = self.attn(inp_V[j, :], inp_V[j, :], inp_V[j, :], mask_V[j, :])
                    att_T, _ = self.attn(inp_T[j, :], inp_T[j, :], inp_T[j, :], mask_T[j, :])
                    att_A, _ = self.attn(inp_A[j, :], inp_A[j, :], inp_A[j, :], mask_A[j, :])
                    new_inp_V[j, :] = att_V + inp_V[j, :]
                    new_inp_A[j, :] = att_A + inp_A[j, :]
                    new_inp_T[j, :] = att_T + inp_T[j, :]
                    #对于每一个moment j ，得到的new_inp_V的每一行代表一个moment的inter-person attention

                # Modality-level intra-personal attention
                att_V, _ = self.attn(new_inp_V[:, k], new_inp_V[:, k], new_inp_V[:, k], mask_V[:, k])
                att_A, _ = self.attn(new_inp_A[:, k], new_inp_A[:, k], new_inp_A[:, k], mask_A[:, k])
                att_T, _ = self.attn(new_inp_T[:, k], new_inp_T[:, k], new_inp_T[:, k], mask_T[:, k])

                # Residual connection
                inner_V = (att_V[target_moment] + new_inp_V[target_moment][k]).squeeze()
                inner_A = (att_A[target_moment] + new_inp_A[target_moment][k]).squeeze()
                inner_T = (att_T[target_moment] + new_inp_T[target_moment][k]).squeeze()

                # Multimodal fusion
                inner_U = self.fusion_layer(torch.cat([inner_V, inner_A, inner_T, inp_P[0][k]]))

                U.append(inner_U)

            if len(U) == 1:
                # Only one character in this sample
                U_all.append(U[0])
            else:
                # Person-level Inter-personal Attention
                U = torch.stack(U, dim=0)
                output, _ = self.attn(U, U, U)
                U = U + output
                U_all.append(U[target_character])

        U_all = torch.stack(U_all, dim=0)
        # Classification
        log_prob = self.out_layer(U_all)
        log_prob = F.log_softmax(log_prob)

        return log_prob

