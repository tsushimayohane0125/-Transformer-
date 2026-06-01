"""
Transformer 英德翻译 - 增强训练版
配置：15000条数据，30轮训练，优化架构
目标：获得更准确的翻译模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from collections import Counter
import time
import os
import random
import warnings
import gc

warnings.filterwarnings('ignore')

# ============================================================
# 1. 设置
# ============================================================
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# 设置CPU线程数
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# 2. 加载数据
# ============================================================
def load_multi30k_local():
    """加载Multi30k数据集"""
    train_en_path = "training/train.en"
    train_de_path = "training/train.de"
    val_en_path = "validation/val.en"
    val_de_path = "validation/val.de"
    test_en_path = "mmt16_task1_test/test.en"
    test_de_path = "mmt16_task1_test/test.de"
    
    def read_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f]
    
    print("正在加载数据集...")
    train_en = read_file(train_en_path)
    train_de = read_file(train_de_path)
    val_en = read_file(val_en_path)
    val_de = read_file(val_de_path)
    test_en = read_file(test_en_path)
    test_de = read_file(test_de_path)
    
    train_pairs = list(zip(train_en, train_de))
    val_pairs = list(zip(val_en, val_de))
    test_pairs = list(zip(test_en, test_de))
    
    print(f"训练集: {len(train_pairs)} 条")
    print(f"验证集: {len(val_pairs)} 条")
    print(f"测试集: {len(test_pairs)} 条")
    
    return train_pairs, val_pairs, test_pairs

# ============================================================
# 3. 构建词表（使用更大词表）
# ============================================================
def build_vocab(sentences, max_size=12000):
    """构建词表 - 使用更大词表提高翻译质量"""
    counter = Counter()
    for sent in sentences:
        counter.update(sent.split())
    
    # 取最常见的词
    sorted_words = sorted(counter.items(), key=lambda x: -x[1])
    sorted_words = sorted_words[:max_size - 4]
    
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab.extend([word for word, _ in sorted_words])
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    
    print(f"词表大小: {len(vocab)}")
    return word2idx, idx2word

# ============================================================
# 4. 数据集类（带缓存加速）
# ============================================================
class CachedTranslationDataset(Dataset):
    """带缓存的高效数据集"""
    def __init__(self, data_pairs, src_word2idx, tgt_word2idx, max_len=45):
        self.data = []
        for src_text, tgt_text in data_pairs:
            src_tokens = src_text.split()[:max_len-2]
            tgt_tokens = tgt_text.split()[:max_len-2]
            
            src_indices = [SOS_IDX] + [src_word2idx.get(t, UNK_IDX) for t in src_tokens] + [EOS_IDX]
            tgt_indices = [SOS_IDX] + [tgt_word2idx.get(t, UNK_IDX) for t in tgt_tokens] + [EOS_IDX]
            
            self.data.append((torch.tensor(src_indices, dtype=torch.long), 
                            torch.tensor(tgt_indices, dtype=torch.long)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded

# ============================================================
# 5. 增强版Transformer模型
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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, padding_idx, smoothing=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        
    def forward(self, output, target):
        vocab_size = output.size(-1)
        confidence = 1.0 - self.smoothing
        log_probs = torch.log_softmax(output, dim=-1)
        
        true_dist = torch.zeros_like(output).fill_(self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), confidence)
        true_dist[:, self.padding_idx] = 0
        
        mask = (target == self.padding_idx).unsqueeze(1)
        true_dist.masked_fill_(mask, 0)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class EnhancedTransformer(nn.Module):
    """增强版Transformer - 更大模型，更好效果"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, 
                 num_layers=3, dim_feedforward=512, dropout=0.2, max_len=50):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer（更大模型）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU激活函数效果更好
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 添加LayerNorm稳定训练
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
    
    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_key_padding_mask = (src == PAD_IDX)
        tgt_key_padding_mask = (tgt == PAD_IDX)
        
        src = self.dropout(self.pos_encoder(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_decoder(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        
        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        output = self.layer_norm(output)
        return self.fc_out(output)
    
    def translate(self, src_tensor, tgt_idx2word, max_len=50):
        """贪心解码翻译"""
        self.eval()
        with torch.no_grad():
            tgt = torch.tensor([[SOS_IDX]], device=src_tensor.device)
            for _ in range(max_len):
                output = self.forward(src_tensor, tgt)
                next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                tgt = torch.cat([tgt, next_token], dim=1)
                if next_token.item() == EOS_IDX:
                    break
        
        tokens = []
        for idx in tgt[0].tolist():
            if idx in [SOS_IDX, EOS_IDX, PAD_IDX]:
                continue
            tokens.append(tgt_idx2word.get(idx, '<unk>'))
        return ' '.join(tokens)
    
    def translate_beam(self, src_tensor, tgt_idx2word, beam_width=3, max_len=50):
        """Beam Search解码（更准确但稍慢）"""
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
        
        tokens = []
        for idx in best_seq:
            if idx in [SOS_IDX, EOS_IDX, PAD_IDX]:
                if idx == EOS_IDX:
                    break
                continue
            tokens.append(tgt_idx2word.get(idx, '<unk>'))
        
        return ' '.join(tokens) if tokens else "【翻译失败】"

# ============================================================
# 6. 增强训练器
# ============================================================
class EnhancedTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
        # 优化器配置（AdamW + 权重衰减）
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0003,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )
        
        # 学习率调度（余弦退火）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        
        # 梯度累积（模拟更大batch）
        self.gradient_accumulation = 2
        self.clip_norm = 1.0
        
        # 学习率预热
        self.warmup_epochs = 3
        self.warmup_lr_start = 1e-5
    
    def get_lr(self, epoch):
        """预热学习率"""
        if epoch < self.warmup_epochs:
            ratio = (epoch + 1) / self.warmup_epochs
            return self.warmup_lr_start + (0.0003 - self.warmup_lr_start) * ratio
        return None
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        self.optimizer.zero_grad()
        
        # 预热
        warmup_lr = self.get_lr(epoch)
        if warmup_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            if epoch == 0:
                print(f"  🌡️ 预热模式: LR从 {self.warmup_lr_start:.2e} 开始")
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = self.model(src, tgt_input)
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss = loss / self.gradient_accumulation
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                print(f"    Batch {batch_idx}/{len(self.train_loader)}, Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs, save_path='enhanced_model.pth'):
        print("\n" + "="*60)
        print("开始增强训练...")
        print(f"训练轮数: {epochs}")
        print(f"预热轮数: {self.warmup_epochs}")
        print(f"梯度累积: {self.gradient_accumulation}")
        print("="*60)
        
        total_start = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # 预热结束后才更新调度器
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # 保存最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                print(f"  ★ 保存最佳模型 (Epoch {epoch+1}, Loss: {val_loss:.4f})")
            
            epoch_time = time.time() - epoch_start
            status = "↓" if is_best else " "
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} {status} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.0f}s")
            
            # 定期清理内存
            if (epoch + 1) % 5 == 0:
                gc.collect()
        
        total_time = time.time() - total_start
        print("="*60)
        print(f"训练完成！总耗时: {total_time/60:.1f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
        print("="*60)
        
        return self.train_losses, self.val_losses

# ============================================================
# 7. 翻译测试程序
# ============================================================
def translate_sentence(model, sentence, src_word2idx, tgt_idx2word, device, use_beam=False):
    """翻译单个句子"""
    src_tokens = sentence.lower().split()
    src_indices = [SOS_IDX] + [src_word2idx.get(t, UNK_IDX) for t in src_tokens] + [EOS_IDX]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
    
    if use_beam:
        return model.translate_beam(src_tensor, tgt_idx2word, beam_width=3)
    else:
        return model.translate(src_tensor, tgt_idx2word)

# ============================================================
# 8. 主程序
# ============================================================
if __name__ == "__main__":
    
    print("="*70)
    print("Transformer 英德翻译 - 增强训练版")
    print("配置: 15000条数据, 30轮训练, 增强模型架构")
    print("预计训练时间: 12-15小时 (取决于CPU性能)")
    print("="*70)
    
    # 加载数据
    try:
        train_pairs, val_pairs, test_pairs = load_multi30k_local()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        exit(1)
    
    # 构建词表（使用更大词表）
    print("\n构建词表中...")
    train_en = [pair[0] for pair in train_pairs]
    train_de = [pair[1] for pair in train_pairs]
    
    src_word2idx, src_idx2word = build_vocab(train_en, max_size=12000)
    tgt_word2idx, tgt_idx2word = build_vocab(train_de, max_size=12000)
    
    # ===== 增强训练配置 =====
    print("\n" + "="*50)
    print("增强训练配置")
    print("="*50)
    
    # 使用15000条数据，30轮训练
    data_size = 15000
    epochs = 30
    batch_size = 64
    
    print(f"\n配置:")
    print(f"  📊 数据量: {data_size} 条")
    print(f"  🔄 训练轮数: {epochs}")
    print(f"  📦 批次大小: {batch_size}")
    print(f"  🧠 模型配置: d_model=256, layers=3, heads=8")
    print(f"  📝 词表大小: 源={len(src_word2idx)}, 目标={len(tgt_word2idx)}")
    print(f"  ⏰ 预计时间: 12-15小时")
    
    # 准备数据集
    print("\n准备数据集...")
    train_subset = train_pairs[:data_size]
    val_subset = val_pairs[:1000]  # 增加验证集大小
    
    train_dataset = CachedTranslationDataset(train_subset, src_word2idx, tgt_word2idx, max_len=45)
    val_dataset = CachedTranslationDataset(val_subset, src_word2idx, tgt_word2idx, max_len=45)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 创建增强模型
    print("\n创建增强模型...")
    model = EnhancedTransformer(
        src_vocab_size=len(src_word2idx),
        tgt_vocab_size=len(tgt_word2idx),
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.2,
        max_len=50
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 损失函数
    criterion = LabelSmoothingLoss(padding_idx=PAD_IDX, smoothing=0.1)
    
    # 训练
    trainer = EnhancedTrainer(model, train_loader, val_loader, criterion, device)
    train_losses, val_losses = trainer.train(epochs, save_path='enhanced_model.pth')
    
    # 保存词表
    torch.save(src_word2idx, 'src_vocab.pth')
    torch.save((tgt_word2idx, tgt_idx2word), 'tgt_vocab.pth')
    print("\n✓ 词表已保存: src_vocab.pth, tgt_vocab.pth")
    
    # 加载最佳模型进行测试
    print("\n" + "="*60)
    print("加载最佳模型进行翻译测试")
    print("="*60)
    
    model.load_state_dict(torch.load('enhanced_model.pth', map_location=device))
    model.eval()
    
    # 测试句子（包括简单和复杂）
    test_sentences = [
        "a man in an orange hat starring at something.",
        "a dog runs fast.",
        "the cat sits on the mat.",
        "a woman is reading a book.",
        "children are playing football.",
        "a young girl is walking her dog in the park.",
        "two men are sitting at a table eating lunch.",
        "the man in the blue suit is talking on his phone."
    ]
    
    print("\n【贪心解码翻译】")
    print("-"*40)
    for i, sentence in enumerate(test_sentences[:5]):
        translation = translate_sentence(model, sentence, src_word2idx, tgt_idx2word, device, use_beam=False)
        print(f"\n{i+1}. EN: {sentence}")
        print(f"   DE: {translation}")
    
    print("\n【Beam Search翻译（更准确）】")
    print("-"*40)
    for i, sentence in enumerate(test_sentences):
        translation = translate_sentence(model, sentence, src_word2idx, tgt_idx2word, device, use_beam=True)
        print(f"\n{i+1}. EN: {sentence}")
        print(f"   DE: {translation}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves (Enhanced Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    best_idx = val_losses.index(min(val_losses))
    plt.plot(best_idx, min(val_losses), 'g*', markersize=15, label=f'Best (Epoch {best_idx+1})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.lr_history, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_curves.png', dpi=150)
    plt.show()
    
    # 总结
    print("\n" + "="*60)
    print("增强训练完成总结")
    print("="*60)
    print(f"  ✓ 数据量: {data_size} 条")
    print(f"  ✓ 训练轮数: {epochs}")
    print(f"  ✓ 最佳验证损失: {min(val_losses):.4f}")
    print(f"  ✓ 模型文件: enhanced_model.pth")
    print(f"  ✓ 词表文件: src_vocab.pth, tgt_vocab.pth")
    print(f"  ✓ 模型参数量: {total_params/1e6:.2f}M")
    print("\n【模型特点】")
    print("  ✓ 更大的模型: d_model=256, layers=3, heads=8")
    print("  ✓ 更大的词表: 12000词")
    print("  ✓ 学习率预热: 前3轮预热")
    print("  ✓ 余弦退火调度: 稳定收敛")
    print("  ✓ 支持Beam Search: 翻译更准确")
    print("="*60)