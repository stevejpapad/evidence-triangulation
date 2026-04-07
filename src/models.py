import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, key_padding_mask=None):

        attn_output, attn_weights = self.mha(query, key_value, key_value, key_padding_mask=key_padding_mask)
        x = self.norm(query + attn_output)
        x = self.norm_ffn(x + self.ffn(x))
        
        return x, attn_weights

        
class TRENT(nn.Module):
    def __init__(self, 
                 emb_dim=768, 
                 evidence_emb=768,
                 hidden_dim=512, 
                 num_heads=8, 
                 dropout=0.1):
        super().__init__()

        self.img_proj = nn.Linear(emb_dim, hidden_dim)
        self.txt_proj = nn.Linear(emb_dim, hidden_dim)
        self.ev_proj  = nn.Linear(evidence_emb, hidden_dim)
        
        self.C_img_ev = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.C_txt_ev = CrossAttentionBlock(hidden_dim, num_heads, dropout)               
        self.C_img_txt = CrossAttentionBlock(hidden_dim, num_heads, dropout)

        fusion_input_dim = 3 * (4 * hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(fusion_input_dim, 1))

    def relational_fusion(self, v1, v2):
        v1 = v1.squeeze(1)
        v2 = v2.squeeze(1)
        
        return torch.cat([v1, v2, torch.abs(v1 - v2), v1 * v2], dim=-1)

    def forward(self, img, txt, i2t_ev, t2t_ev):
            """
            img, txt: (B, 768)
            i2t_ev:   (B, M, 768)
            t2t_ev:   (B, M, 768)
            """

            # projections
            if i2t_ev.dim() == 2:
                i2t_ev = i2t_ev.unsqueeze(1)
                
            if t2t_ev.dim() == 2:
                t2t_ev = t2t_ev.unsqueeze(1)
                                                                           
            z_img = self.img_proj(img).unsqueeze(1)    
            z_txt = self.txt_proj(txt).unsqueeze(1)    

            all_ev = torch.cat([i2t_ev, t2t_ev], dim=1) 
            z_ev   = self.ev_proj(all_ev)  
        
            key_padding_mask = (all_ev.abs().sum(dim=-1) == 0)
            key_padding_mask[key_padding_mask.all(dim=1), 0] = False           

            # cross-attention 
            c_img_ev, _ = self.C_img_ev(z_img, z_ev, key_padding_mask=key_padding_mask)
            c_txt_ev, _ = self.C_txt_ev(z_txt, z_ev, key_padding_mask=key_padding_mask)
            c_img_txt, _  = self.C_img_txt(z_txt, z_img)

            # relational fusion
            z1 = self.relational_fusion(z_img, c_img_ev)
            z2 = self.relational_fusion(z_txt, c_txt_ev)
            z3 = self.relational_fusion(z_txt, c_img_txt)

            # classification
            z_final = torch.cat([z1, z2, z3], dim=-1) 
        
            logits = self.classifier(z_final).flatten()

            return logits                                                          
                                                 
