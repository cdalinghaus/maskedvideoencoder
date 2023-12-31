import torch.nn as nn
import torch.nn.functional as F
import torch
from helpers import plotframes, patchify, unpatchify
import random

"""
# Hyperparameters
NUM_FRAMES = 10
COLOR_CHANNELS = 3
IMG_SIZE = 256
PATCH_SIZE = 16

D_DIM = 128
EMBEDDING_DIM = 32
PATCH_KEEP_PERCENTAGE = 0.5

NUM_PATCHES = int((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE))*NUM_FRAMES

LEARNING_RATE = 1e-4
"""

class MaskedVideoTransformer(nn.Module):
    def __init__(self,
        NUM_FRAMES = 10,
        COLOR_CHANNELS = 3,
        IMG_SIZE = 128,
        PATCH_SIZE = 16,
        D_DIM = 128,
        EMBEDDING_DIM = 32,
        PATCH_KEEP_PERCENTAGE = 0.8,
        ENC_HEADS = 12,
        ENC_LAYERS = 12,
        ENC_FF = 2048,
        DEC_HEADS = 12,
        DEC_LAYERS = 12,
        DEC_FF = 2048,
    ):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        NUM_PATCHES = int((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE))*NUM_FRAMES

        self.NUM_PATCHES = NUM_PATCHES
        self.NUM_FRAMES = NUM_FRAMES
        self.COLOR_CHANNELS = COLOR_CHANNELS
        self.IMG_SIZE = IMG_SIZE
        self.PATCH_SIZE = PATCH_SIZE
        self.D_DIM = D_DIM
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.PATCH_KEEP_PERCENTAGE = PATCH_KEEP_PERCENTAGE

        # Helper tensors registered as pytorch buffer
        self.register_buffer("zero_tensor", torch.Tensor([0]).long())
        self.register_buffer("embedding_indices", torch.arange(NUM_PATCHES))

        # Define the Model
        self.P = torch.nn.Linear(PATCH_SIZE*PATCH_SIZE*COLOR_CHANNELS, D_DIM)
        self.P_invert = torch.nn.Linear(D_DIM, PATCH_SIZE*PATCH_SIZE*COLOR_CHANNELS)

        self.Embedding = torch.nn.Embedding(NUM_PATCHES, EMBEDDING_DIM)
        #self.Missing_Patch_Embedding = torch.nn.Embedding(1, D_DIM)
        self.Missing_Patch_Embedding = torch.nn.Embedding(NUM_PATCHES, D_DIM)

        print(D_DIM + EMBEDDING_DIM, D_DIM, EMBEDDING_DIM)
        encoder_layer = torch.nn.TransformerEncoderLayer(D_DIM + EMBEDDING_DIM, ENC_HEADS, dim_feedforward=ENC_FF, dropout=0.1)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, ENC_LAYERS)

        decoder_layer = torch.nn.TransformerEncoderLayer(D_DIM + EMBEDDING_DIM, DEC_HEADS, dim_feedforward=DEC_FF, dropout=0.1)
        self.transformer_decoder = torch.nn.TransformerEncoder(decoder_layer, DEC_LAYERS)

    def forward(self, X, target=None):
        # 1. Create, flatten and project the patches
        patches_per_dim = self.IMG_SIZE // self.PATCH_SIZE
        batch_size = X.shape[0]
        #print("X.shape", X.shape)
        
        patches = patchify(X, self.PATCH_SIZE)
        
        patches_shape_unflattened = patches.shape
        patches_flattened = torch.flatten(patches, start_dim=2)
        #print("flattened", patches_flattened.shape)
        patches_projected = self.P(patches_flattened)

        batch_size, num_patches, _ = patches_projected.shape

        # 2. Add positional embeddings to the patches
        embedded_vectors = self.Embedding(self.embedding_indices)
        embedded_vectors = embedded_vectors.repeat(batch_size, 1, 1)
        X = torch.cat([patches_projected, embedded_vectors], dim=-1)

        # 3. Randomly blind n percent of the patches
        num_patches = X.shape[1]
        patches_to_keep = [random.random() < self.PATCH_KEEP_PERCENTAGE for _ in range(num_patches)]
        patches_to_keep_inverted = [not x for x in patches_to_keep]
        X_for_encoder = X[:, patches_to_keep].clone()
        X_blinded_for_display = patches.clone()
        X_blinded_for_display[:, patches_to_keep_inverted] = 1
        X_blinded_for_display = unpatchify(X_blinded_for_display, self.IMG_SIZE, self.PATCH_SIZE, self.NUM_FRAMES, self.COLOR_CHANNELS)
        X_for_decoder = X.clone()
        X_for_decoder[:, patches_to_keep_inverted] = 0

        # 4. Actually use the ENCODER
        representation_after_encoder = self.transformer_encoder(X_for_encoder)

        # 5. Mix the encoder representation into the decoder input (which is blinded)
        index_pointer = 0
        embedded_vectors = self.Embedding(self.embedding_indices)
        
        for idx, keep in enumerate(patches_to_keep):
            if keep:
                X_for_decoder[:, idx] = representation_after_encoder[:, index_pointer]
                index_pointer += 1
            else:
                #print(learnable_missing_patch_token[0].shape, embedded_vectors[idx].shape)
                _device = self.dummy_param.device
                learnable_missing_patch_token = self.Missing_Patch_Embedding(torch.LongTensor([idx]).to(_device))
                X_for_decoder[:, idx] = torch.concatenate([learnable_missing_patch_token[0], embedded_vectors[idx]])

        # 6. Actually use the DECODER
        decoder_representation = self.transformer_decoder(X_for_decoder)

        # 7. project and reshape to original size and shape
        decoder_representation = decoder_representation[:, :, :self.D_DIM]
        decoder_representation = self.P_invert(decoder_representation)
        #decoder_representation = torch.sigmoid(decoder_representation)
        decoder_representation = decoder_representation.reshape(patches_shape_unflattened)
        
        decoder_patches = unpatchify(decoder_representation, self.IMG_SIZE, self.PATCH_SIZE, self.NUM_FRAMES, self.COLOR_CHANNELS)

        if target is not None:
            decoder_representation[:, patches_to_keep_inverted] = 0
            target_patches = patchify(target, self.PATCH_SIZE)
            #print("target patches shape", target_patches.shape)
            target_patches[:, patches_to_keep_inverted] = 0
            return decoder_patches, (X_blinded_for_display, decoder_representation, target_patches)

        return decoder_patches, (X_blinded_for_display, )
