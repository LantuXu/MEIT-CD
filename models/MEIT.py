import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d

class TokenScorer(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, 1)

    def forward(self, source_tokens, target_tokens):
        """
        source_tokens: (B, N_src, D) - as query
        target_tokens: (B, N_tgt, D) - as key and value
        """
        B, N_src, D = source_tokens.shape
        N_tgt = target_tokens.shape[1]

        q = self.q(source_tokens)  # (B, N_src, D)
        k = self.k(target_tokens)  # (B, N_tgt, D)
        v = self.v(target_tokens)  # (B, N_tgt, 1)

        # **Global Cross Attention Calculation**
        attn_scores = torch.einsum('bnd,bmd->bnm', q, k) / (D ** 0.5)  # (B, N_src, N_tgt)
        attn_weights = attn_scores.softmax(dim=-1)  # (B, N_src, N_tgt)

        # Weighted value gets score
        scores = torch.einsum('bnm,bmd->bnd', attn_weights, v).squeeze(-1)  # (B, N_src)

        return scores  # (B, N_src)

class MDecoder(nn.Module):
    def __init__(self, input_channels, embed_dim, num_heads, ff_dim, num_layers,heads=8, num_partitions=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=1)
        self.layernorm_proj = nn.LayerNorm(embed_dim)
        self.score_layer = TokenScorer(embed_dim)
        self.num_partitions = num_partitions

        self.transformers = nn.ModuleList([
            TransformerDecoder(dim=embed_dim, depth=num_layers, heads=heads, dim_head=num_heads,
                               mlp_dim=ff_dim, dropout=dropout, softmax=True)
            for _ in range(num_partitions)
        ])
        self.final_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, m):
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        tokens = x_proj.flatten(2).transpose(1, 2)
        tokens = self.layernorm_proj(tokens)
        scores = self.score_layer(tokens,m).squeeze(-1)

        partition_size = tokens.shape[1] // self.num_partitions
        sorted_indices = torch.argsort(scores, dim=1, descending=True)

        partition_scores = []  # Store the average score of each partition as weight

        for i in range(self.num_partitions):
            partition_indices = sorted_indices[:,
                                i * partition_size: (i + 1) * partition_size]  # [B, N//num_partitions]
            partition_score = torch.gather(scores, 1, partition_indices).mean(dim=1)  # [B]
            partition_scores.append(partition_score)

        partition_scores = torch.stack(partition_scores, dim=1)  # [B, num_partitions]
        partition_weights = torch.softmax(partition_scores, dim=1)  # [B, num_partitions]

        output = torch.zeros_like(tokens)

        for i in range(self.num_partitions):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < self.num_partitions - 1 else tokens.shape[1]
            selected_indices = sorted_indices[:, start_idx:end_idx]
            selected_tokens = torch.gather(tokens, 1, selected_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
            processed = self.transformers[i](selected_tokens, m)
            weight = partition_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            processed = processed * weight

            output.scatter_(1, selected_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]), processed)

        output = self.final_layernorm(output)
        return output


class DualBranchResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=False):
        """
        Double branch residual network
        """
        super(DualBranchResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet1 = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            self.resnet2 = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet1 = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            self.resnet2 = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet1 = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            self.resnet2 = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=128, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred1 = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.conv_pred2 = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 ,x3, x5 = self.forward_single1(x1)
        x2 ,x4, x5= self.forward_single2(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single1(self, x):
        # resnet layers
        x = self.resnet1.conv1(x)
        x = self.resnet1.bn1(x)
        x_2 = self.resnet1.relu(x)
        x = self.resnet1.maxpool(x_2)

        x_4 = self.resnet1.layer1(x)
        x_8 = self.resnet1.layer2(x_4)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet1.layer3(x_8)
        if self.resnet_stages_num == 5:
            x_8 = self.resnet1.layer4(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred1(x)
        return x,x_4,x_2
    def forward_single2(self, x):
        # resnet layers
        x = self.resnet2.conv1(x)
        x = self.resnet2.bn1(x)
        x_2 = self.resnet2.relu(x)
        x = self.resnet2.maxpool(x_2)

        x_4 = self.resnet2.layer1(x)
        x_8 = self.resnet2.layer2(x_4)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet2.layer3(x_8)
        if self.resnet_stages_num == 5:
            x_8 = self.resnet2.layer4(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred2(x)
        return x,x_4,x_2

class MEIT(DualBranchResNet):
    """
    Dual-Branch Residual Feature Extraction  + MT1 Model + Skip Transformer Fusion + A Small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=False,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(MEIT, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a_0 = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.conv_a_1 = nn.Conv2d(64, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.conv_a_2 = nn.Conv2d(64, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.conv_a_3 = nn.Conv2d(128, self.token_len*2, kernel_size=1,
                                padding=0, bias=False)
        self.conv_a_4 = nn.Conv2d(128, self.token_len*2, kernel_size=1,
                                padding=0, bias=False)
        self.conv_up = TwoLayerConv2d(in_channels=32, out_channels=64,)
        self.conv_up1 = TwoLayerConv2d(in_channels=128, out_channels=64, )
        self.tokenizer = tokenizer
        if not self.tokenizer:
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
            self.pos_embedding1 = nn.Parameter(torch.randn(1, self.token_len*2, 64))
            self.pos_embedding2 = nn.Parameter(torch.randn(1, self.token_len*2, 64))
            self.pos_embedding3 = nn.Parameter(torch.randn(1, self.token_len*2, 128))
            self.pos_embedding4 = nn.Parameter(torch.randn(1, self.token_len*2, 128))
        decoder_pos_size = 256//4
        decoder_pos_size1 = 256//2
        decoder_pos_size2 = 256
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,decoder_pos_size,decoder_pos_size))
            self.pos_embedding_decoder1 = nn.Parameter(torch.randn((1,64,decoder_pos_size1,decoder_pos_size1)))#128
            self.pos_embedding_decoder2 = nn.Parameter(torch.randn((1,64,decoder_pos_size2,decoder_pos_size2)))#256
            self.pos_embedding_decoder3 = nn.Parameter(torch.randn((1,128, decoder_pos_size1, decoder_pos_size1)))
            self.pos_embedding_decoder4 = nn.Parameter(torch.randn((1,128, decoder_pos_size2, decoder_pos_size2)))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer_0 = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_1 = Transformer(dim=dim*2, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head*2,
                                       mlp_dim=mlp_dim*2, dropout=0)
        self.transformer_2 = Transformer(dim=dim*2, depth=self.enc_depth, heads=8,
                                        dim_head=self.dim_head*2,
                                        mlp_dim=mlp_dim*2, dropout=0)
        self.transformer_3 = Transformer(dim=dim*4, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head*4,
                                       mlp_dim=mlp_dim*4, dropout=0)
        self.transformer_4 = Transformer(dim=dim*4, depth=self.enc_depth, heads=8,
                                        dim_head=self.dim_head*4,
                                        mlp_dim=mlp_dim*4, dropout=0)

        self.transformer_decoder_0 = MDecoder(input_channels=dim, embed_dim=dim,
                                                                        num_heads=self.decoder_dim_head,
                                                                        ff_dim=mlp_dim,
                                                                        num_layers=self.dec_depth,
                                                                        num_partitions=1, dropout=0)
        self.transformer_decoder_1 = MDecoder(input_channels=64,embed_dim=64,
                                                                        num_heads=self.decoder_dim_head*2,
                                                                        ff_dim=mlp_dim*2,
                                                                        num_layers=self.dec_depth//2,
                                                                        num_partitions=4,dropout=0)
        self.transformer_decoder_2 = MDecoder(input_channels=64, embed_dim=64,
                                                                        num_heads=self.decoder_dim_head*2,
                                                                        ff_dim=mlp_dim*2,
                                                                        num_layers=self.dec_depth //4,
                                                                        num_partitions=16, dropout=0)
        self.transformer_decoder_3 = MDecoder(input_channels=128, embed_dim=128,
                                                                         num_heads=self.decoder_dim_head*4,
                                                                         ff_dim=mlp_dim*4,
                                                                         num_layers=self.dec_depth//2,
                                                                         num_partitions=4, dropout=0)
        self.transformer_decoder_4 = MDecoder(input_channels=128, embed_dim=128,
                                                                         num_heads=self.decoder_dim_head*4,
                                                                         ff_dim=mlp_dim*4,
                                                                         num_layers=self.dec_depth // 4,
                                                                         num_partitions=16, dropout=0)

    def _forward_semantic_tokens_0(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a_0(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_semantic_tokens_1(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a_1(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_semantic_tokens_2(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a_2(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_semantic_tokens_3(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a_3(x)
        spatial_attention = spatial_attention.view([b, self.token_len*2, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_semantic_tokens_4(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a_4(x)
        spatial_attention = spatial_attention.view([b, self.token_len*2, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer_0(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer_0(x)
        return x

    def _forward_transformer_1(self, x):
        if self.with_pos:
            x += self.pos_embedding1
        x = self.transformer_1(x)
        return x

    def _forward_transformer_2(self, x):
        if self.with_pos:
            x += self.pos_embedding2
        x = self.transformer_2(x)
        return x

    def _forward_transformer_3(self, x):
        if self.with_pos:
            x += self.pos_embedding3
        x = self.transformer_3(x)
        return x

    def _forward_transformer_4(self, x):
        if self.with_pos:
            x += self.pos_embedding4
        x = self.transformer_4(x)
        return x

    def _forward_transformer_decoder_0(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = self.transformer_decoder_0(x,m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_transformer_decoder_1(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder1
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder1
        x = self.transformer_decoder_1(x,m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_transformer_decoder_2(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder2
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder2
        x = self.transformer_decoder_2(x,m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_transformer_decoder_3(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder3
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder3
        x = self.transformer_decoder_3(x,m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_transformer_decoder_4(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder4
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder4
        x = self.transformer_decoder_4(x,m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1,x3,x5 = self.forward_single1(x1)
        x2,x4,x6 = self.forward_single2(x2)

        # MT1 processing
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
            token3 = self._forward_semantic_tokens_new1(x3)
            token4 = self._forward_semantic_tokens_new1(x4)
            token5 = self._forward_semantic_tokens_new2(x5)
            token6 = self._forward_semantic_tokens_new2(x6)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
            token3 = self._forward_reshape_tokens(x3)
            token4 = self._forward_reshape_tokens(x4)
            token5 = self._forward_reshape_tokens(x5)
            token6 = self._forward_reshape_tokens(x6)
        # # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer0(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
            self.tokens_ = torch.cat([token3, token4], dim=1)
            self.tokens = self._forward_transformer1(self.tokens_)
            token3, token4 = self.tokens.chunk(2, dim=1)
            self.tokens_ = torch.cat([token5, token6], dim=1)
            self.tokens = self._forward_transformer2(self.tokens_)
            token5, token6 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder_new0(x1,token1)
            x2 = self._forward_transformer_decoder_new0(x2,token2)
            x3 = self._forward_transformer_decoder_new1(x3,token3)
            x4 = self._forward_transformer_decoder_new1(x4,token4)
            x5 = self._forward_transformer_decoder_new2(x5,token5)
            x6 = self._forward_transformer_decoder_new2(x6,token6)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
            x3 = self._forward_simple_decoder(x3, token3)
            x4 = self._forward_simple_decoder(x4, token4)
            x5 = self._forward_simple_decoder(x5, token5)
            x6 = self._forward_simple_decoder(x6, token6)

        x_end = torch.abs(x1 - x2)
        x7 = torch.abs(x3 - x4)
        x8 = torch.abs(x5 - x6)
        #skip transform fusion(MT2)
        x_end= self.conv_up(x_end)
        x_end = self.upsamplex2(x_end)
        x_end = torch.cat([x_end, x7], dim=1)

        if self.tokenizer:
            token7 = self._forward_semantic_tokens_new3(x_end)
            token7 = self._forward_transformer3(token7)
        if self.with_decoder:
            x_end = self._forward_transformer_decoder_new3(x_end, token7)

        x_end = self.conv_up1(x_end)
        x_end = self.upsamplex2(x_end)
        x_end = torch.concat([x_end,x8], dim=1)

        if self.tokenizer:
            token8 = self._forward_semantic_tokens_new4(x_end)
        if self.with_decoder:
            x_end = self._forward_transformer_decoder_new4(x_end, token8)
        x = self.upsamplex2(x_end)

        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    B, C, H, W = 2, 3, 512, 512
    input_tensor1 = torch.randn(B, C, H, W)
    input_tensor2 = torch.randn(B, C, H, W)

    model=MEIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                    with_pos='learned', enc_depth=1, dec_depth=8)

    out = model(input_tensor1, input_tensor2)
    print("Final output shape:", out.shape)
