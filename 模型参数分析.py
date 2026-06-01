"""
增强版Transformer模型参数分析程序
分析 enhanced_model.pth 的各项参数指标
"""

import torch
import torch.nn as nn
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（解决图表中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模型定义（与训练时完全一致）
# ============================================================
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

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
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embedding层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)
    
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

# ============================================================
# 2. 参数分析器
# ============================================================
class ModelParameterAnalyzer:
    def __init__(self, model_path='enhanced_model.pth'):
        self.model_path = model_path
        self.model = None
        self.analysis_results = {}
        
    def load_model(self):
        """加载模型"""
        print("="*70)
        print("加载模型...")
        print("="*70)
        
        # 先加载词表获取大小
        if os.path.exists('src_vocab.pth'):
            src_word2idx = torch.load('src_vocab.pth', map_location='cpu')
            tgt_data = torch.load('tgt_vocab.pth', map_location='cpu')
            src_vocab_size = len(src_word2idx)
            tgt_vocab_size = len(tgt_data[0])
            print(f"从词表文件获取: 源语言词表={src_vocab_size}, 目标语言词表={tgt_vocab_size}")
        else:
            # 从模型文件推断
            state_dict = torch.load(self.model_path, map_location='cpu')
            src_vocab_size = state_dict['encoder_embedding.weight'].shape[0]
            tgt_vocab_size = state_dict['decoder_embedding.weight'].shape[0]
            print(f"从模型文件推断: 源语言词表={src_vocab_size}, 目标语言词表={tgt_vocab_size}")
        
        # 创建模型
        self.model = EnhancedTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        )
        
        # 加载权重
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ 成功加载模型: {self.model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {self.model_path}")
            return False
        
        return True
    
    def count_parameters(self):
        """详细统计参数量"""
        print("\n" + "="*70)
        print("1. 模型总参数量分析")
        print("="*70)
        
        total_params = 0
        layer_params = defaultdict(int)
        
        for name, param in self.model.named_parameters():
            params = param.numel()
            total_params += params
            
            # 分类统计
            if 'encoder_embedding' in name or 'decoder_embedding' in name:
                layer_params['embedding'] += params
            elif 'pos_encoder' in name or 'pos_decoder' in name:
                layer_params['positional_encoding'] += params
            elif 'fc_out' in name:
                layer_params['output_layer'] += params
            elif 'layer_norm' in name:
                layer_params['layer_norm'] += params
            elif 'self_attn' in name or 'multihead_attn' in name:
                layer_params['attention'] += params
            elif 'linear1' in name or 'linear2' in name:
                layer_params['ffn'] += params
            else:
                layer_params['other'] += params
        
        # 计算总参数量
        self.analysis_results['total_params'] = total_params
        self.analysis_results['layer_params'] = layer_params
        
        print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"\n各组件参数量分布:")
        print("-" * 50)
        for layer, params in sorted(layer_params.items(), key=lambda x: -x[1]):
            percentage = params / total_params * 100
            print(f"  {layer:20s}: {params:>12,} ({percentage:>5.1f}%)")
        
        return layer_params
    
    def analyze_embedding(self):
        """分析Embedding层"""
        print("\n" + "="*70)
        print("2. Embedding层参数量分析")
        print("="*70)
        
        encoder_embed_params = self.model.encoder_embedding.weight.numel()
        decoder_embed_params = self.model.decoder_embedding.weight.numel()
        total_embed_params = encoder_embed_params + decoder_embed_params
        
        print(f"\nEncoder Embedding层:")
        print(f"  形状: {self.model.encoder_embedding.weight.shape}")
        print(f"  参数量: {encoder_embed_params:,} ({encoder_embed_params/1e6:.2f}M)")
        
        print(f"\nDecoder Embedding层:")
        print(f"  形状: {self.model.decoder_embedding.weight.shape}")
        print(f"  参数量: {decoder_embed_params:,} ({decoder_embed_params/1e6:.2f}M)")
        
        print(f"\nEmbedding层总计:")
        print(f"  参数量: {total_embed_params:,} ({total_embed_params/1e6:.2f}M)")
        print(f"  占模型比例: {total_embed_params/self.analysis_results['total_params']*100:.1f}%")
        
        # 分析词向量统计
        with torch.no_grad():
            enc_weight = self.model.encoder_embedding.weight
            dec_weight = self.model.decoder_embedding.weight
            
            print(f"\n词向量统计:")
            print(f"  Encoder权重范围: [{enc_weight.min():.3f}, {enc_weight.max():.3f}]")
            print(f"  Encoder权重均值: {enc_weight.mean():.4f}")
            print(f"  Encoder权重标准差: {enc_weight.std():.4f}")
            print(f"  Decoder权重范围: [{dec_weight.min():.3f}, {dec_weight.max():.3f}]")
            print(f"  Decoder权重均值: {dec_weight.mean():.4f}")
            print(f"  Decoder权重标准差: {dec_weight.std():.4f}")
        
        return {'encoder': encoder_embed_params, 'decoder': decoder_embed_params, 'total': total_embed_params}
    
    def analyze_attention(self):
        """分析Multi-Head Attention层"""
        print("\n" + "="*70)
        print("3. Multi-Head Attention参数量分析")
        print("="*70)
        
        d_model = self.model.d_model
        nhead = self.model.nhead
        num_layers = self.model.num_layers
        
        # 单层Attention的参数量
        # Q, K, V 投影: 3 * d_model * d_model
        # 输出投影: d_model * d_model
        # 总: 4 * d_model * d_model
        single_attn_params = 4 * d_model * d_model
        
        # Encoder Self-Attention
        encoder_attn_params = num_layers * single_attn_params
        
        # Decoder Self-Attention + Cross-Attention
        decoder_attn_params = 2 * num_layers * single_attn_params  # Self + Cross
        
        total_attn_params = encoder_attn_params + decoder_attn_params
        
        print(f"\n配置信息:")
        print(f"  d_model: {d_model}")
        print(f"  nhead: {nhead}")
        print(f"  层数: {num_layers}")
        print(f"  单层Attention参数量: {single_attn_params:,}")
        
        print(f"\nAttention参数量分解:")
        print(f"  Encoder Self-Attention ({num_layers}层): {encoder_attn_params:,} ({encoder_attn_params/1e6:.2f}M)")
        print(f"  Decoder Self-Attention ({num_layers}层): {decoder_attn_params//2:,} ({decoder_attn_params/2/1e6:.2f}M)")
        print(f"  Decoder Cross-Attention ({num_layers}层): {decoder_attn_params//2:,} ({decoder_attn_params/2/1e6:.2f}M)")
        
        print(f"\nAttention总计:")
        print(f"  参数量: {total_attn_params:,} ({total_attn_params/1e6:.2f}M)")
        
        return {'total': total_attn_params, 'per_layer': single_attn_params}
    
    def analyze_ffn(self):
        """分析Feed Forward Network层"""
        print("\n" + "="*70)
        print("4. Feed Forward Network参数量分析")
        print("="*70)
        
        d_model = self.model.d_model
        dim_feedforward = self.model.dim_feedforward
        num_layers = self.model.num_layers
        
        # 单层FFN参数量: 两个线性层
        # 第一层: d_model * dim_feedforward
        # 第二层: dim_feedforward * d_model
        single_ffn_params = 2 * d_model * dim_feedforward
        
        # Encoder FFN
        encoder_ffn_params = num_layers * single_ffn_params
        
        # Decoder FFN
        decoder_ffn_params = num_layers * single_ffn_params
        
        total_ffn_params = encoder_ffn_params + decoder_ffn_params
        
        print(f"\n配置信息:")
        print(f"  d_model: {d_model}")
        print(f"  dim_feedforward: {dim_feedforward}")
        print(f"  层数: {num_layers}")
        print(f"  单层FFN参数量: {single_ffn_params:,} ({single_ffn_params/1e6:.2f}M)")
        
        print(f"\nFFN参数量分解:")
        print(f"  Encoder FFN ({num_layers}层): {encoder_ffn_params:,} ({encoder_ffn_params/1e6:.2f}M)")
        print(f"  Decoder FFN ({num_layers}层): {decoder_ffn_params:,} ({decoder_ffn_params/1e6:.2f}M)")
        
        print(f"\nFFN总计:")
        print(f"  参数量: {total_ffn_params:,} ({total_ffn_params/1e6:.2f}M)")
        
        return {'total': total_ffn_params, 'per_layer': single_ffn_params}
    
    def analyze_encoder_decoder(self):
        """分析Encoder和Decoder的参数组成"""
        print("\n" + "="*70)
        print("5. Encoder和Decoder的参数组成")
        print("="*70)
        
        d_model = self.model.d_model
        num_layers = self.model.num_layers
        dim_feedforward = self.model.dim_feedforward
        
        # 单层Encoder组成
        single_encoder_params = {
            'self_attention': 4 * d_model * d_model,  # Q,K,V,O投影
            'attention_norm': 2 * d_model,             # LayerNorm参数
            'ffn': 2 * d_model * dim_feedforward,      # 两个线性层
            'ffn_norm': 2 * d_model                    # LayerNorm参数
        }
        single_encoder_total = sum(single_encoder_params.values())
        
        # 单层Decoder组成
        single_decoder_params = {
            'self_attention': 4 * d_model * d_model,
            'self_attn_norm': 2 * d_model,
            'cross_attention': 4 * d_model * d_model,
            'cross_attn_norm': 2 * d_model,
            'ffn': 2 * d_model * dim_feedforward,
            'ffn_norm': 2 * d_model
        }
        single_decoder_total = sum(single_decoder_params.values())
        
        # 总参数
        total_encoder_params = num_layers * single_encoder_total
        total_decoder_params = num_layers * single_decoder_total
        
        print(f"\n单层Encoder参数组成:")
        print("-" * 60)
        for component, params in single_encoder_params.items():
            print(f"  {component:20s}: {params:>12,} ({params/single_encoder_total*100:.1f}%)")
        print(f"  {'单层Encoder总计':20s}: {single_encoder_total:>12,}")
        
        print(f"\n单层Decoder参数组成:")
        print("-" * 60)
        for component, params in single_decoder_params.items():
            print(f"  {component:20s}: {params:>12,} ({params/single_decoder_total*100:.1f}%)")
        print(f"  {'单层Decoder总计':20s}: {single_decoder_total:>12,}")
        
        print(f"\n总参数组成 ({num_layers}层):")
        print("-" * 60)
        print(f"  Encoder总参数量: {total_encoder_params:,} ({total_encoder_params/1e6:.2f}M)")
        print(f"  Decoder总参数量: {total_decoder_params:,} ({total_decoder_params/1e6:.2f}M)")
        print(f"  Encoder+Decoder总计: {(total_encoder_params+total_decoder_params):,} ({(total_encoder_params+total_decoder_params)/1e6:.2f}M)")
        
        return {
            'per_encoder': single_encoder_total,
            'per_decoder': single_decoder_total,
            'total_encoder': total_encoder_params,
            'total_decoder': total_decoder_params
        }
    
    def compare_scales(self):
        """比较不同模型规模下参数量的变化"""
        print("\n" + "="*70)
        print("6. 不同模型规模下参数量的变化")
        print("="*70)
        
        # 定义不同规模的模型配置
        scales = [
            {'name': '极小 (Tiny)', 'd_model': 64, 'layers': 1, 'ffn': 128, 'vocab': 8000},
            {'name': '小 (Small)', 'd_model': 128, 'layers': 2, 'ffn': 256, 'vocab': 8000},
            {'name': '中 (Medium)', 'd_model': 256, 'layers': 3, 'ffn': 512, 'vocab': 10000},
            {'name': '大 (Large)', 'd_model': 384, 'layers': 4, 'ffn': 768, 'vocab': 12000},
            {'name': '超大 (XL)', 'd_model': 512, 'layers': 6, 'ffn': 1024, 'vocab': 15000},
            {'name': '当前模型', 'd_model': 256, 'layers': 3, 'ffn': 512, 'vocab': 12000, 'current': True}
        ]
        
        results = []
        for cfg in scales:
            # 计算参数量
            embed_params = (cfg['vocab'] * cfg['d_model']) * 2  # encoder + decoder
            attn_per_layer = 4 * cfg['d_model'] * cfg['d_model']
            ffn_per_layer = 2 * cfg['d_model'] * cfg['ffn']
            
            encoder_layer = attn_per_layer + ffn_per_layer
            decoder_layer = 2 * attn_per_layer + ffn_per_layer
            
            total_params = (embed_params + 
                           cfg['layers'] * (encoder_layer + decoder_layer) +
                           cfg['vocab'] * cfg['d_model'])  # output layer
            
            results.append({
                'name': cfg['name'],
                'd_model': cfg['d_model'],
                'layers': cfg['layers'],
                'ffn': cfg['ffn'],
                'vocab': cfg['vocab'],
                'total': total_params,
                'embed': embed_params,
                'attention': 3 * cfg['layers'] * attn_per_layer,
                'ffn_total': 2 * cfg['layers'] * ffn_per_layer,
                'is_current': cfg.get('current', False)
            })
        
        print(f"\n{'模型规模':<12} {'d_model':<8} {'层数':<6} {'词表':<8} {'总参数量':<15} {'Embedding':<12} {'Attention':<12} {'FFN':<12}")
        print("-" * 95)
        for r in results:
            marker = " ← 当前" if r['is_current'] else ""
            print(f"{r['name']:<12} {r['d_model']:<8} {r['layers']:<6} {r['vocab']:<8} {r['total']/1e6:<14.2f}M {r['embed']/1e6:<11.2f}M {r['attention']/1e6:<11.2f}M {r['ffn_total']/1e6:<11.2f}M{marker}")
        
        return results
    
    def analyze_relationships(self):
        """分析参数量与训练时间、显存、模型效果的关系"""
        print("\n" + "="*70)
        print("7. 参数量与训练时间、显存占用、模型效果之间的关系")
        print("="*70)
        
        total_params = self.analysis_results['total_params']
        d_model = self.model.d_model
        num_layers = self.model.num_layers
        
        # 训练时间估算（基于当前模型）
        # 假设每秒处理100个token，每句话约10个词
        tokens_per_second = 100  # CPU保守估计
        data_size = 15000
        seq_len = 20
        epochs = 30
        
        # 计算FLOPs估算
        flops_per_token = (12 * d_model * d_model * num_layers)  # 简化估算
        total_flops = data_size * seq_len * epochs * flops_per_token
        
        # 显存估算
        # 模型参数 + 梯度 + 优化器状态 + 激活值
        param_memory = total_params * 4 / 1024 / 1024  # 4 bytes per param -> MB
        grad_memory = param_memory  # 梯度同等大小
        optimizer_memory = param_memory * 2  # Adam优化器状态
        activation_memory = 256  # 激活值估算
        
        total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
        
        print(f"\n【当前模型配置】")
        print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  d_model: {d_model}")
        print(f"  层数: {num_layers}")
        
        print(f"\n【训练时间估算】")
        print(f"  数据量: 15000条")
        print(f"  训练轮数: 30轮")
        print(f"  预计总FLOPs: {total_flops/1e12:.2f}T")
        print(f"  CPU训练时间(保守): 约12-15小时")
        print(f"  GPU训练时间(如V100): 约2-3小时")
        
        print(f"\n【显存占用估算】")
        print(f"  模型参数: {param_memory:.1f} MB")
        print(f"  梯度: {grad_memory:.1f} MB")
        print(f"  优化器状态(Adam): {optimizer_memory:.1f} MB")
        print(f"  激活值: {activation_memory:.1f} MB")
        print(f"  总计: {total_memory:.1f} MB (~{total_memory/1024:.2f} GB)")
        
        print(f"\n【模型效果关系】")
        print(f"  参数量与性能关系: 对数增长")
        print(f"  当前模型预期BLEU分数: 25-30")
        print(f"  效果瓶颈: 训练数据量(15000条)和训练时长")
        
        print(f"\n【参数量与性能关系公式】")
        print(f"  验证损失 ≈ {2.5:.1f} - {0.3:.1f} × log(参数量/1e6)")
        print(f"  BLEU分数 ≈ {15:.0f} + {8:.0f} × log(参数量/1e6)")
        
        print(f"\n【不同参数量模型的效果预估】")
        print(f"  {'参数量':<12} {'验证损失':<12} {'BLEU分数':<12} {'训练时间(CPU)':<15}")
        print("-" * 55)
        sizes = [1e6, 5e6, 10e6, 25e6, 50e6, 100e6]
        for size in sizes:
            loss = 2.8 - 0.3 * math.log(size/1e6)
            bleu = 15 + 8 * math.log(size/1e6)
            time_hours = size/1e6 * 4  # 粗略估算
            marker = " ← 当前" if abs(size - total_params) < 5e6 else ""
            print(f"{size/1e6:>6.0f}M      {loss:.2f}         {bleu:.0f}           {time_hours:.0f}h{marker}")
    
    def plot_analysis(self, scale_results):
        """绘制分析图表"""
        print("\n" + "="*70)
        print("生成分析图表...")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 当前模型参数分布饼图
        ax1 = axes[0, 0]
        layer_params = self.analysis_results['layer_params']
        labels = list(layer_params.keys())
        sizes = list(layer_params.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('当前模型参数分布')
        
        # 2. 不同规模参数量对比
        ax2 = axes[0, 1]
        names = [r['name'] for r in scale_results]
        totals = [r['total']/1e6 for r in scale_results]
        colors_bar = ['#ff9999' if not r.get('is_current', False) else '#66b3ff' for r in scale_results]
        bars = ax2.bar(names, totals, color=colors_bar)
        ax2.set_xlabel('模型规模')
        ax2.set_ylabel('参数量 (M)')
        ax2.set_title('不同规模模型总参数量对比')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, totals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # 3. 各组件参数占比
        ax3 = axes[1, 0]
        components = ['Embedding', 'Attention', 'FFN', 'Output', 'Norm', 'Other']
        comp_sizes = [
            self.analysis_results['layer_params'].get('embedding', 0),
            self.analysis_results['layer_params'].get('attention', 0),
            self.analysis_results['layer_params'].get('ffn', 0),
            self.analysis_results['layer_params'].get('output_layer', 0),
            self.analysis_results['layer_params'].get('layer_norm', 0),
            self.analysis_results['layer_params'].get('other', 0)
        ]
        colors_comp = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#f0c2c2']
        ax3.pie(comp_sizes, labels=components, colors=colors_comp, autopct='%1.1f%%', startangle=90)
        ax3.set_title('各组件参数占比')
        
        # 4. 参数量与性能关系
        ax4 = axes[1, 1]
        sizes_log = [math.log(r['total']) for r in scale_results]
        # 模拟性能曲线
        performance = [15 + 8 * math.log(r['total']/1e6) for r in scale_results]
        ax4.plot([r['total']/1e6 for r in scale_results], performance, 'bo-', linewidth=2, markersize=8)
        ax4.set_xlabel('参数量 (M)')
        ax4.set_ylabel('预估BLEU分数')
        ax4.set_title('参数量与模型效果关系')
        ax4.grid(True, alpha=0.3)
        # 标记当前模型
        current_total = self.analysis_results['total_params']/1e6
        current_perf = 15 + 8 * math.log(current_total)
        ax4.plot(current_total, current_perf, 'r*', markersize=15, label=f'当前模型: {current_perf:.0f} BLEU')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('enhanced_model_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ 图表已保存: enhanced_model_analysis.png")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("="*70)
        print("增强版Transformer模型参数分析报告")
        print("="*70)
        
        # 加载模型
        if not self.load_model():
            return
        
        # 1. 总参数量
        self.count_parameters()
        
        # 2. Embedding层
        self.analyze_embedding()
        
        # 3. Attention层
        self.analyze_attention()
        
        # 4. FFN层
        self.analyze_ffn()
        
        # 5. Encoder/Decoder组成
        self.analyze_encoder_decoder()
        
        # 6. 不同规模对比
        scale_results = self.compare_scales()
        
        # 7. 关系分析
        self.analyze_relationships()
        
        # 8. 绘制图表
        self.plot_analysis(scale_results)
        
        # 总结
        print("\n" + "="*70)
        print("分析总结")
        print("="*70)
        total_params = self.analysis_results['total_params']
        print(f"""
当前模型: enhanced_model.pth
├── 模型配置: d_model=256, layers=3, heads=8, ffn=512
├── 总参数量: {total_params:,} ({total_params/1e6:.2f}M)
├── Embedding占比: {self.analysis_results['layer_params'].get('embedding', 0)/total_params*100:.1f}%
├── Attention占比: {self.analysis_results['layer_params'].get('attention', 0)/total_params*100:.1f}%
├── FFN占比: {self.analysis_results['layer_params'].get('ffn', 0)/total_params*100:.1f}%
├── 预估训练时间: 12-15小时 (CPU)
├── 预估显存占用: {total_params*4/1024/1024*4:.0f} MB
└── 预估BLEU分数: {15 + 8 * math.log(total_params/1e6):.0f}
""")
        
        return self.analysis_results

# ============================================================
# 3. 主程序
# ============================================================
if __name__ == "__main__":
    
    print("="*70)
    print("Transformer 增强模型参数分析程序")
    print("分析模型: enhanced_model.pth")
    print("="*70)
    
    # 检查模型文件
    if not os.path.exists('enhanced_model.pth'):
        print("\n⚠️ 未找到 enhanced_model.pth")
        print("请先运行 train_enhanced.py 训练模型")
        print("或修改模型路径")
        exit(1)
    
    # 运行分析
    analyzer = ModelParameterAnalyzer(model_path='enhanced_model.pth')
    results = analyzer.run_full_analysis()