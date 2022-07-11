import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, img_encoder, nl_encoder, output_size, hidden_size = 768, alignment_size = 1024,
                 src_length = 64, dropout_rate = 0.25):
        super().__init__()
        
        self.img_encoder = img_encoder
        self.nl_encoder = nl_encoder
        self.img_linear = nn.Sequential(
            nn.Linear(alignment_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, output_size)
        )
        
        self.nl_linear = nn.Sequential(
            nn.Linear(2 * alignment_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, output_size)
        )
        
        self.combined_linear = nn.Sequential(
            nn.Linear(5 * alignment_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, output_size)
        )
        
        self.Wl = nn.Linear(64, 1)
        
        self.attn = Attention()
        # self.attn = BiAttn(2*alignment_size)
        self.alignment = nn.Sequential(
            nn.AvgPool2d(8, 1)
        )
        # self.decide_weight = nn.parameter.Parameter(data=torch.rand(3, 1),requires_grad=True)
        # self.softmax = nn.Softmax(0)
        self.layernorm = nn.LayerNorm(alignment_size)
        
        self.multiheadattn_im = MultiheadAttn(alignment_size, alignment_size, alignment_size)
        self.multiheadattn_nl = MultiheadAttn(alignment_size, alignment_size, alignment_size)
        # self.ngramcnn = NgramCNN(src_length, hidden_size * 2, output_size, dropout_rate = dropout_rate)
        # self.ngramcnn1 = NgramCNN(src_length, hidden_size * 1, output_size, dropout_rate = dropout_rate)
        self.gru = BIGRUDecoder(hidden_size, alignment_size // 2)
        
        
        
    def forward(self, img = None, src_ids = None, src_mask = None):
        img_out, encoder_output, out = None, None, None
        if img != None:
            img_out = self.img_encoder(img)                                   # 16, 1024, 8, 8
            
            # image logits
            shape = img_out.shape
            img_only = img_out.contiguous().view(shape[0], shape[1], -1).permute([0,2,1])
            img_only = self.layernorm(img_only + self.multiheadattn_im(img_only)).permute([0,2,1])
            img_only = self.alignment(img_only.view(shape)).squeeze()
            
        if src_ids != None and src_mask != None: 
            encoder_output = self.nl_encoder(src_ids, src_mask).last_hidden_state   # 16, 64, 768
            encoder_output = self.gru(encoder_output)                               # 16, 64, 1024
            
            # text logits
            nl_only = self.layernorm(encoder_output + self.multiheadattn_nl(encoder_output))
            nl_only = torch.cat((nl_only[:,0,:], nl_only[:,-1,:]), dim = -1)
            
            
        if img_out != None and encoder_output == None:
            img_logits = self.img_linear(img_only)
            logits = img_logits
        elif img_out == None and encoder_output != None:
            nl_logits = self.nl_linear(nl_only)
            logits = nl_logits
        else:
            # combined logits
            img_hid = img_out.contiguous().view(img_out.size(0),img_out.size(1),-1).permute([0,2,1])
            nl_hid = encoder_output

            img_attn = self.attn(nl_hid, img_hid, nl_hid)
            nl_attn = self.attn(img_hid, nl_hid, img_hid)
            hid = self.Wl(torch.cat((img_attn, nl_attn), dim = -1).permute([0,2,1])).squeeze(2)
            combined_logits = self.combined_linear(torch.cat((hid, nl_only, img_only), dim = -1))
            """
            logits = torch.cat((
                    img_logits.unsqueeze(2),
                    nl_logits.unsqueeze(2),
                    combined_logits.unsqueeze(2)
                ),dim = -1).matmul(self.softmax(self.decide_weight)).squeeze()
            """
            
            logits = combined_logits
                
            
        return logits

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        loss = self.loss_func(logits, labels)
        return loss
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v):
        e = torch.einsum('bxh,byh->bxy', k, q)
        a = self.softmax(e)
        out = torch.einsum('byh,bxy->bxh', v, a)
        
        return out

class BiAttn(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.softmax = nn.Softmax(dim = -1)
        self.score_func = nn.Linear(input_size, 1)
    def forward(self, q, k, v):
        B, X, H = k.size()
        B, Y, H = q.size()
        
        k_expand = k.unsqueeze(2).repeat(1,1,Y,1)
        q_expand = q.unsqueeze(1).repeat(1,X,1,1)
        
        # B,X,Y
        e = self.score_func(torch.cat((k_expand, q_expand), dim = -1)).squeeze()
        a = self.softmax(e)
        
        out = torch.einsum('byh,bxy->bxh', v, a)
        return out
        

class NgramCNN(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size,
                 n_span = [2, 3, 4], num_filters = 64, dropout_rate = 0.3):
        super().__init__()
        self.cnn_list = []
        for n in n_span:
            self.cnn_list += [nn.Sequential(
                nn.Conv2d(1, num_filters, (n, hidden_size)),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.MaxPool2d((seq_len - n + 1, 1), 1)
            ).cuda()] * 2

        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(n_span) * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, output_size)
        )
        
    def forward(self, x):
        features = []
        for conv in self.cnn_list:
            features += [conv(x.unsqueeze(1)).view(x.size(0), -1)]
        features = torch.cat(features, dim = -1)
        logits = self.linear(features)
        return logits
    
class BIGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate = 0.3):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first = True, bidirectional = True)
    
    def forward(self, x):
        l = self.gru(x)[0]
        return l
        
class MultiheadAttn(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads = 2):
        super().__init__()
        
        assert (hidden_size % num_heads == 0) and (proj_size % num_heads == 0)
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, proj_size)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(0.1)
        self.num_heads = num_heads
    
    def forward(self, x):
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        hidden_size = q.size(-1)
        q = q.contiguous().view(x.size(0), x.size(1), self.num_heads, hidden_size // self.num_heads)
        k = k.contiguous().view(x.size(0), x.size(1), self.num_heads, hidden_size // self.num_heads)
        v = v.contiguous().view(x.size(0), x.size(1), self.num_heads, hidden_size // self.num_heads)
        
        z = self.softmax(torch.einsum('bsnh,btnh->bnts', q, k) / (hidden_size // self.num_heads) ** 0.5)
        z = self.dropout(z)
        out = torch.einsum('bnts,btnh->bsnh', z, v).contiguous().view(x.size(0), x.size(1), -1)
        out = self.Wo(out)
        
        return out
        
                
        