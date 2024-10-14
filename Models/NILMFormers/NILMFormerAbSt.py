import torch
import torch.nn as nn

from Models.NILMFormers.Layers.Transformer import EncoderLayer
from Models.NILMFormers.Layers.ConvLayer import DilatedBlock

# ======================= NILMFormer stats mechanisms impact =======================#
class NILMFormerAbStats(nn.Module):
    def __init__(self,
                 instance_norm=True, revin=False, add_token_stat=True, learn_stats=True, 
                 c_in=1, c_embedding=8, c_out=1, n_encoder_layers=3, 
                 d_model=128, ratio_pe=4,
                 kernel_size=3, kernel_size_head=3, dilations=[1, 2, 4, 8], conv_bias=True,
                 att_param={'att_mask_diag': True, 'att_mask_flag': False, 'att_learn_scale': False,
                            'activation': 'gelu', 'pffn_ratio': 4, 'n_head': 8, 
                            'prenorm': True, 'norm': "LayerNorm", 
                            'store_att': False, 'dp_rate': 0.2, 'attn_dp_rate': 0.2}):
        """
        NILMFormer model use for the ablation study conducted in Section 5.4.1 : Non-stationarity consideration

        Default configuration is equivalent to NILMFormer.
        """
        super().__init__()

        assert d_model%ratio_pe==0, f'd_model need to be divisble by ratio_pe: {ratio_pe}.'
        assert instance_norm!=revin, f'Cannot apply revin and Instance Norm.'
  
        self.d_model       = d_model
        self.c_in          = c_in
        self.c_out         = c_out

        self.instance_norm  = instance_norm
        self.revin          = revin
        self.add_token_stat = add_token_stat
        self.learn_stats    = learn_stats
                
        #============ Embedding ============#
        ie = ratio_pe - 1
        d_model_ = int(ie * d_model//ratio_pe)
        # Resnet block with dilation parameters
        self.EmbedBlock = DilatedBlock(c_in=c_in, c_out=d_model_, kernel_size=kernel_size, dilation_list=dilations, bias=conv_bias)
        # Embedding proj
        self.ProjEmbedding = nn.Conv1d(in_channels=c_embedding, out_channels=d_model//ratio_pe, kernel_size=1)

        if self.instance_norm:
            self.ProjStats1 = nn.Linear(2, d_model)
            self.ProjStats2 = nn.Linear(d_model, 2)

        if self.revin:
            self.affine_weight = nn.Parameter(torch.ones(1))
            self.affine_bias   = nn.Parameter(torch.zeros(1))
            self.eps = 1e-5
        
       #============ Encoder ============#
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(d_model, 
                                       d_ff=d_model * att_param['pffn_ratio'], n_heads=att_param['n_head'], 
                                       dp_rate=att_param['dp_rate'], attn_dp_rate=att_param['attn_dp_rate'], 
                                       att_mask_diag=att_param['att_mask_diag'], 
                                       att_mask_flag=att_param['att_mask_flag'], 
                                       learnable_scale=att_param['att_learn_scale'], 
                                       store_att=att_param['store_att'],  
                                       norm=att_param['norm'], prenorm=att_param['prenorm'], 
                                       activation=att_param['activation']))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = torch.nn.Sequential(*layers)

        #============ Downstream Task Head ============#
        self.DownstreamTaskHead = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=kernel_size_head, padding=kernel_size_head//2, padding_mode='replicate')

        #============ Initializing weights ============#        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def freeze_params(self, model_part, rq_grad=False):
        for _, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)
    
    def forward(self, x) -> torch.Tensor:
        # Input as B 1+e L 
        # Separate load curve and embedding
        encoding = x[:, 1:, :] # B 1 L
        x        = x[:, :1, :] # B e L

        # === Instance Normalization === #
        if self.instance_norm:
            inst_mean = torch.mean(x, dim=-1, keepdim=True).detach() # Mean: B 1 1
            inst_std  = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6).detach() # STD: B 1 1

            x = (x - inst_mean) / inst_std # Instance z-norm: B 1 1

            if self.revin:
                x = x.permute(0,2,1)
                x = x * self.affine_weight
                x = x + self.affine_bias
                x = x.permute(0,2,1)
        
        # === Embedding === # 
        # Conv Dilated Embedding Block for aggregate
        x = self.EmbedBlock(x) # B L D-D/4
        # Conv1x1 for encoded time features
        encoding = self.ProjEmbedding(encoding) # B L D/4
        # Concat Token from aggregate and encoded time features
        x = torch.cat([x, encoding], dim=1).permute(0, 2, 1) # B L D

        # === Mean and Std projection === #
        if self.instance_norm:
            stats_token = self.ProjStats1(torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)) # B 1 D
            if self.add_token_stat:
                x = torch.cat([x, stats_token], dim=1) # Add stats token B L+1 D

        # === Forward Transformer Encoder === #
        x = self.forward_encoder(x) 
        if self.instance_norm:
            if self.add_token_stat:
                x = x[:, :-1, :] # Remove stats token: B L D

        # === Conv Head === #
        x = x.permute(0, 2, 1) # B D L
        x = self.DownstreamTaskHead(x) # B 1 L

        # === Reverse Instance Normalization === #
        if self.instance_norm:
            if self.learn_stats:
                # Proj back stats_token stats to get mean' and std' if learnable stats
                stats_out    = self.ProjStats2(stats_token) # B 1 2
                outinst_mean = stats_out[:, :, 0].unsqueeze(1) # B 1 1
                outinst_std  = stats_out[:, :, 1].unsqueeze(1) # B 1 1

                x = x * outinst_mean + outinst_std
            else:
                if self.revin:
                    x = x.permute(0,2,1)
                    x = x - self.affine_bias
                    x = x / (self.affine_weight + self.eps*self.eps)
                    x = x.permute(0,2,1)
                x = x * inst_mean + inst_std

        return x