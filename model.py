# model.py
import torch # type: ignore
import torch.nn as nn # type: ignore


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=(530, 450), patch_size=(4, 10, 15),window_size=12, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size[1]) * (img_size[1] // patch_size[2]) * (window_size // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input x shape: (B, T, C, H, W) -> e.g., (4, 12, 3, 616, 760)
        # Conv3d expects: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Output of Conv3d: (B, embed_dim, T_patches, H_patches, W_patches)
        x = self.proj(x)
        
        # Flatten and transpose to Transformer's expected shape: (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, img_size=(530, 450), patch_size=(4, 10, 15),window_size=12, in_chans=3, out_chans=1,
                 num_predictions=1, embed_dim=768, depth=6, num_heads=8,
                 mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.num_predictions = num_predictions
        self.patch_embed = PatchEmbed3D(img_size, patch_size, window_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

       
        self.head = nn.Linear(embed_dim, num_predictions * out_chans * img_size[0] * img_size[1])
        self.output_shape = (num_predictions, out_chans, img_size[0], img_size[1])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer_encoder(x)
        cls_output = x[:, 0]
        prediction = self.head(cls_output)
        prediction = prediction.view(B, *self.output_shape)
        return prediction