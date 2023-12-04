from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math

import torch
from torch.nn import Linear, Sequential, ModuleList, Transformer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, optim
import torch.nn as nn
import numpy as np
from load_raw_data import load_data


BATCH_SIZE = 1
TRAIN_SIZE = 10000
TEST_SIZE = 11673
SRC_VOCAB_SIZE = 1000
TGT_VOCAB_SIZE = 4            # Num of bonds
EMB_SIZE = 32
NHEAD = 2
FFN_HID_DIM = 256
BATCH_SIZE = 1
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
SRC_SEQ_LEN = 1703 # 94 + 1600
ATOM_NODE_FEAT_SIZE = 6
NUM_ATOM_TYPES = 47
DEVICE = "cpu" # DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Dataset(Dataset):
    def __init__(self, element_properties, mol_atom_node_feats, mol_atom_valency, adjacency_matrices, msp_in):
        self.element_properties = element_properties
        # self.mol_atom_list = mol_atom_list
        self.mol_atom_node_feats = mol_atom_node_feats
        self.mol_atom_valency = mol_atom_valency
        self.adjacency_matrices = adjacency_matrices
        # self.mol_in = mol_in
        self.msp_in = msp_in
        # self.atom_nodes_in = atom_nodes_in
        # self.y = torch.argmax(y, axis=-1)
        # self.y = replace_eos_with_pad_idx(self.y, pad_idx=PAD_IDX)
        
    def __getitem__(self, index):
        return self.mol_atom_node_feats[index], self.mol_atom_valency[index], self.adjacency_matrices[index], self.msp_in[index]
    
    def __len__(self):
        return len(self.msp_in)

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2GraphTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 num_atom_types: int,
                 atom_node_in_feat_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2GraphTransformer, self).__init__()
        self.emb_size = emb_size
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.msp_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.atom_node_emb = TokenEmbedding(num_atom_types, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.atom_nodes_linear = nn.Linear(atom_node_in_feat_size, emb_size)
        self.bond_in_decoder_linear_atoms = nn.Linear(2 * emb_size, emb_size)
        self.bond_in_decoder_linear_edges = nn.Linear(3 * emb_size, emb_size)
        self.bond_in_decoder_linear = nn.Linear(2 * emb_size, emb_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                atom_node_in: Tensor,
                num_atoms_in_mol: list,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.msp_tok_emb(src))
        atom_node_emb = self.positional_encoding(self.atom_nodes_linear(atom_node_in.float()))
        src_emb = torch.cat([src_emb, atom_node_emb], dim=1)

        adjacency_matrices_emb = self.tgt_tok_emb(trg)
        mol_num_atoms = trg.shape[-1]
        lower_tri_indices = torch.tril_indices(row=mol_num_atoms, col=mol_num_atoms, offset=-1)
        for b in range(BATCH_SIZE):
            atom_pairs = torch.cat([atom_node_emb[b, lower_tri_indices[0]],
                                    atom_node_emb[b, lower_tri_indices[1]]], dim=-1)
            atom_pairs = self.bond_in_decoder_linear_atoms(atom_pairs.unsqueeze(0))
            mol_edges = torch.cat([atom_node_emb[b, lower_tri_indices[0]],
                                   atom_node_emb[b, lower_tri_indices[1]],
                                   adjacency_matrices_emb[b, lower_tri_indices[0], lower_tri_indices[1]]], dim=-1)
            mol_edges = self.bond_in_decoder_linear_edges(mol_edges.unsqueeze(0))
            # right shift mol_edges
            mol_edges = torch.cat([torch.zeros(1, 1, self.emb_size).to(DEVICE), mol_edges[:, :-1]], dim=1)
            # print("mol_edges train shape:", mol_edges.shape)
            tgt_emb = torch.cat([mol_edges, atom_pairs], dim=-1)
            tgt_emb = self.bond_in_decoder_linear(tgt_emb)
        outs = self.transformer(src_emb.transpose(0, 1), tgt_emb.transpose(0, 1), src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, atom_node_in: Tensor, src_mask: Tensor):
        src_emb = self.positional_encoding(self.msp_tok_emb(src))
        atom_node_emb = self.positional_encoding(self.atom_nodes_linear(atom_node_in.float()))
        src_emb = torch.cat([src_emb, atom_node_emb], dim=1).transpose(0, 1)
        return self.transformer.encoder(src_emb, src_mask), atom_node_emb

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(tgt, memory, tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, src_seq_len=None, tgt_seq_len=None):
    if src_seq_len == None:
        src_seq_len = src.shape[1]
    if tgt_seq_len == None:
        tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1).to(DEVICE)
    temp_src_padding_mask = torch.zeros((src_seq_len, src.shape[0])).type(torch.bool).to(DEVICE)
    temp_src_padding_mask[-src_padding_mask.shape[0]: , ] = src_padding_mask
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).to(DEVICE)
    return src_mask, tgt_mask, temp_src_padding_mask, tgt_padding_mask


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) 
    train_loss, correct, count_atoms = 0, 0, 0
    for batch, (batch_mol_atom_node_feats, batch_mol_atom_valency, batch_adjacency_matrices, batch_msp_in) in enumerate(dataloader):
        # Compute prediction and loss
        mol_num_atoms = batch_adjacency_matrices.shape[-1]
        if mol_num_atoms > 4:
            tgt_mask = generate_square_subsequent_mask(mol_num_atoms * (mol_num_atoms - 1) // 2)

            out = model.forward(src=batch_msp_in.to(DEVICE), trg=batch_adjacency_matrices.to(DEVICE),
                                atom_node_in=batch_mol_atom_node_feats.to(DEVICE),
                                num_atoms_in_mol=[batch_mol_atom_valency.shape[-1]],
                                src_mask=None, tgt_mask=tgt_mask,
                                src_padding_mask=None, tgt_padding_mask=None,
                                memory_key_padding_mask=None)
            try:
                pred_adj_mat = torch.zeros((mol_num_atoms, mol_num_atoms, 4)).to(DEVICE)
                pred_adj_mat[torch.tril_indices(mol_num_atoms, mol_num_atoms, offset=-1).tolist()] = out[0]
                pred_adj_mat = pred_adj_mat + pred_adj_mat.transpose(0, 1)
                # print(pred_adj_mat.shape, batch_adjacency_matrices.shape)
                loss = loss_fn(pred_adj_mat.reshape(mol_num_atoms*mol_num_atoms, 4),
                               batch_adjacency_matrices[0].reshape(mol_num_atoms*mol_num_atoms).to(DEVICE))
                train_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except:
                print(batch_adjacency_matrices.shape)
                print(out.shape)
            # print(pred_adj_mat.shape, batch_adjacency_matrices[0].shape)
            correct += (pred_adj_mat.argmax(-1).to("cpu") == batch_adjacency_matrices[0]).sum()
            count_atoms += mol_num_atoms * mol_num_atoms

            if batch % 4000 == 0:
                loss, current = loss.item(), (batch + 1) * len(batch_adjacency_matrices)
                print(f"loss: {train_loss/current:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Accuracy: ", correct/count_atoms, "Train Loss:", train_loss/size)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset) 
    test_loss, correct, count_atoms = 0, 0, 0
    model.eval()
    for batch, (batch_mol_atom_node_feats, batch_mol_atom_valency, batch_adjacency_matrices, batch_msp_in) in enumerate(dataloader):
        # Compute prediction and loss
        mol_num_atoms = batch_adjacency_matrices.shape[-1]
        if mol_num_atoms > 4:
            memory, atom_node_emb = model.encode(src=batch_msp_in.to(DEVICE),
                                                 atom_node_in=batch_mol_atom_node_feats.to(DEVICE),
                                                 src_mask=None)
            lower_tri_indices = torch.tril_indices(row=mol_num_atoms, col=mol_num_atoms, offset=-1)
            atom_pairs = torch.cat([atom_node_emb[0, lower_tri_indices[0]],
                                    atom_node_emb[0, lower_tri_indices[1]]], dim=-1)
            atom_pairs = model.bond_in_decoder_linear_atoms(atom_pairs.unsqueeze(0))
            mol_edges = torch.zeros((1, 1, model.emb_size)).to(DEVICE)
            tgt_emb = torch.cat([mol_edges, atom_pairs[:, :1]], dim=-1)
            tgt_emb = model.bond_in_decoder_linear(tgt_emb)
            out = model.decode(tgt=tgt_emb, memory=memory,
                               tgt_mask = generate_square_subsequent_mask(1))
            out = model.generator(out)
            ys = out.argmax(-1)
            for i in range(2, mol_num_atoms * (mol_num_atoms - 1) // 2 + 1):
                mol_edges = torch.cat([atom_node_emb[0, lower_tri_indices[0, :i-1]],
                                       atom_node_emb[0, lower_tri_indices[1, :i-1]],
                                       model.tgt_tok_emb(ys[0])], dim=-1)
                mol_edges = model.bond_in_decoder_linear_edges(mol_edges.unsqueeze(0))
                mol_edges = torch.cat([torch.zeros((BATCH_SIZE, 1, model.emb_size)).to(DEVICE), mol_edges], dim=1)
                tgt_emb = torch.cat([mol_edges, atom_pairs[:, :i]], dim=-1)
                tgt_emb = model.bond_in_decoder_linear(tgt_emb)
                out = model.decode(tgt=tgt_emb.transpose(0, 1), memory=memory,
                                   tgt_mask=None)
                out = model.generator(out)
                ys = out.argmax(-1).reshape(1, i)
            pred_adj_mat = torch.zeros((mol_num_atoms, mol_num_atoms, 4)).to(DEVICE)
            pred_adj_mat[torch.tril_indices(mol_num_atoms, mol_num_atoms, offset=-1).tolist()] = out.transpose(0, 1)[0]
            pred_adj_mat = pred_adj_mat + pred_adj_mat.transpose(0, 1)
            try:
                loss = loss_fn(pred_adj_mat.reshape(mol_num_atoms*mol_num_atoms, 4),
                               batch_adjacency_matrices[0].reshape(mol_num_atoms*mol_num_atoms).to(DEVICE))
                test_loss += loss
            except:
                print(batch_adjacency_matrices.shape)
                print(out.shape)
            # print(pred_adj_mat.shape, batch_adjacency_matrices[0].shape)
            correct += (pred_adj_mat.argmax(-1).to("cpu") == batch_adjacency_matrices[0]).sum()
            count_atoms += mol_num_atoms * mol_num_atoms
            if batch % 4000 == 0:
                loss, current = loss.item(), (batch + 1) * len(batch_adjacency_matrices)
                print(f"loss: {test_loss/current:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Accuracy: ", correct/count_atoms, "Train Loss:", test_loss/size)


data_df, element_properties, mol_atom_node_feats, mol_atom_valency, adjacency_matrices = load_data()

train_data = Dataset(element_properties=element_properties, 
                     mol_atom_node_feats=mol_atom_node_feats[ :TRAIN_SIZE],
                     mol_atom_valency=mol_atom_valency[ :TRAIN_SIZE],
                     adjacency_matrices=adjacency_matrices[ :TRAIN_SIZE],
                     msp_in=torch.from_numpy(np.array(list(data_df["msp_seq"][ :TRAIN_SIZE]))))

# test_data = Dataset(element_properties=element_properties[TRAIN_SIZE: ],
#                     mol_atom_node_feats=mol_atom_node_feats[TRAIN_SIZE: ],
#                     mol_atom_valency=mol_atom_valency[TRAIN_SIZE: ],
#                     adjacency_matrices=adjacency_matrices[TRAIN_SIZE: ],
#                     msp_in=torch.from_numpy(np.array(list(data_df["msp_seq"][TRAIN_SIZE: ]))))

test_data = Dataset(element_properties=element_properties[TRAIN_SIZE: TRAIN_SIZE+TEST_SIZE],
                    mol_atom_node_feats=mol_atom_node_feats[TRAIN_SIZE: TRAIN_SIZE+TEST_SIZE],
                    mol_atom_valency=mol_atom_valency[TRAIN_SIZE: TRAIN_SIZE+TEST_SIZE],
                    adjacency_matrices=adjacency_matrices[TRAIN_SIZE: TRAIN_SIZE+TEST_SIZE],
                    msp_in=torch.from_numpy(np.array(list(data_df["msp_seq"][TRAIN_SIZE: TRAIN_SIZE+TEST_SIZE]))))

# test_data = Dataset(element_properties=element_properties[-TEST_SIZE: ],
#                     mol_atom_node_feats=mol_atom_node_feats[-TEST_SIZE: ],
#                     mol_atom_valency=mol_atom_valency[-TEST_SIZE: ],
#                     adjacency_matrices=adjacency_matrices[-TEST_SIZE: ],
#                     msp_in=torch.from_numpy(np.array(list(data_df["msp_seq"][-TEST_SIZE: ]))))

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(temp_msp_in, 
#                                                                      temp_adjacency_matrices[0, lower_tri_indices[0], lower_tri_indices[1]].unsqueeze(0), 
#                                                                      src_seq_len=SRC_SEQ_LEN)

torch.manual_seed(0)
transformer = Seq2GraphTransformer(num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                                   emb_size=EMB_SIZE, num_atom_types=NUM_ATOM_TYPES, atom_node_in_feat_size=ATOM_NODE_FEAT_SIZE, nhead=NHEAD,
                                   src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE, dim_feedforward=FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss() # weight=torch.tensor([2.9717e-05, 1.5397e-04, 9.1281e-04, 2.3764e-02])
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# train_loop(train_dataloader, transformer, loss_fn, optimizer)
# test_loop(train_dataloader, transformer, loss_fn)
for i in range(20):
    print("Epoch", i+1)
    train_loop(train_dataloader, transformer, loss_fn, optimizer)
    # torch.save(transformer.state_dict(), "transformer_models/Seq2Graph_40000_epoch_"+str(i+1)+".pth")
    test_loop(test_dataloader, transformer, loss_fn)
