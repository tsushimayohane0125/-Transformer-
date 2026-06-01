"""
增强版翻译程序 - 使用训练好的模型
"""

import torch
import torch.nn as nn
import math
import os
import warnings
warnings.filterwarnings('ignore')

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 模型定义（与训练时一致）
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EnhancedTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.2, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_key_padding_mask = (src == PAD_IDX)
        tgt_key_padding_mask = (tgt == PAD_IDX)
        
        src = self.dropout(self.pos_encoder(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_decoder(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        
        output = self.transformer(src, tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask)
        
        output = self.layer_norm(output)
        return self.fc_out(output)
    
    def translate(self, src_tensor, tgt_idx2word, max_len=50):
        self.eval()
        with torch.no_grad():
            tgt = torch.tensor([[SOS_IDX]], device=src_tensor.device)
            for _ in range(max_len):
                output = self.forward(src_tensor, tgt)
                next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == EOS_IDX:
                    break
        
        tokens = [tgt_idx2word.get(idx, '<unk>') for idx in tgt[0].tolist() 
                  if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
        return ' '.join(tokens)
    
    def translate_beam(self, src_tensor, tgt_idx2word, beam_width=3, max_len=50):
        self.eval()
        if src_tensor.dim() == 1:
            src_tensor = src_tensor.unsqueeze(0)
        
        with torch.no_grad():
            beams = [([SOS_IDX], 0.0)]
            for _ in range(max_len):
                all_candidates = []
                for seq, score in beams:
                    if seq[-1] == EOS_IDX:
                        all_candidates.append((seq, score))
                        continue
                    if len(seq) >= max_len:
                        all_candidates.append((seq + [EOS_IDX], score))
                        continue
                    
                    tgt_tensor = torch.tensor([seq], device=src_tensor.device)
                    output = self.forward(src_tensor, tgt_tensor)
                    log_probs = torch.log_softmax(output[0, -1, :], dim=-1)
                    top_k = min(beam_width, log_probs.size(-1))
                    top_log_probs, top_indices = log_probs.topk(top_k)
                    
                    for i in range(top_k):
                        token = top_indices[i].item()
                        log_prob = top_log_probs[i].item()
                        all_candidates.append((seq + [token], score + log_prob))
                
                if not all_candidates:
                    break
                beams = sorted(all_candidates, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
                if all(seq[-1] == EOS_IDX for seq, _ in beams):
                    break
            
            best_seq = max(beams, key=lambda x: x[1] / len(x[0]))[0]
        
        tokens = [tgt_idx2word.get(idx, '<unk>') for idx in best_seq 
                  if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
        return ' '.join(tokens) if tokens else "【翻译失败】"

# ============================================================
# 翻译器
# ============================================================
class Translator:
    def __init__(self, model_path='enhanced_model.pth', use_beam=True):
        print("加载翻译模型...")
        
        # 加载词表
        self.src_word2idx = torch.load('src_vocab.pth', map_location='cpu')
        tgt_data = torch.load('tgt_vocab.pth', map_location='cpu')
        self.tgt_word2idx, self.tgt_idx2word = tgt_data
        
        print(f"源语言词表: {len(self.src_word2idx)}")
        print(f"目标语言词表: {len(self.tgt_word2idx)}")
        
        # 创建模型
        self.model = EnhancedTransformer(
            src_vocab_size=len(self.src_word2idx),
            tgt_vocab_size=len(self.tgt_word2idx)
        ).to(device)
        
        # 加载权重
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ 加载模型成功: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
        
        self.model.eval()
        self.use_beam = use_beam
    
    def translate(self, text):
        src_tokens = text.lower().split()
        src_indices = [SOS_IDX] + [self.src_word2idx.get(w, UNK_IDX) for w in src_tokens] + [EOS_IDX]
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
        
        if self.use_beam:
            return self.model.translate_beam(src_tensor, self.tgt_idx2word)
        else:
            return self.model.translate(src_tensor, self.tgt_idx2word)
    
    def interactive(self):
        print("\n" + "="*50)
        print("交互式翻译模式")
        print(f"解码方式: {'Beam Search' if self.use_beam else '贪心'}")
        print("输入 'quit' 退出")
        print("="*50)
        
        while True:
            text = input("\n英文: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text:
                continue
            
            import time
            start = time.time()
            translation = self.translate(text)
            elapsed = time.time() - start
            
            print(f"德语: {translation}")
            print(f"耗时: {elapsed*1000:.0f}ms")

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='enhanced_model.pth')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--greedy', action='store_true', help='使用贪心解码（更快）')
    
    args = parser.parse_args()
    
    translator = Translator(model_path=args.model, use_beam=not args.greedy)
    
    if args.text:
        print(f"\n原文: {args.text}")
        print(f"译文: {translator.translate(args.text)}")
    else:
        translator.interactive()