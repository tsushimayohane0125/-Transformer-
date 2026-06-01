基于 Attention is all you need 实现英德语言翻译的代码复现和模型参数分析

目录：
1.项目简介
2.小组介绍
3.环境要求
4.文件目录
5.数据集收集
6.模型架构
7.训练过程
8.实验结果
9.模型参数统计
10.参考项目和文献


1.项目简介

参考链接和论文：https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/README.md
实验的主要目的：通过原文中transformer模型进行对英语德语的复现，并提供数据集，模型的训练等分析工具
数据集收集约15000条，进行30轮训练。
模型架构：基于transfermer编码器-解码器的架构进行训练增强

2，小组介绍

模型、参数收集训练以及部分代码复现：廖罡艺
PPT 海报 论文解读：王子豪
代码复现、论文解读、文件整理和总结：莫宗霖

3，环境要求

Python 3.8+
PyTorch 1.9+
matplotlib, numpy, collections, warnings, gc, random, time, os
推荐运行环境：CPU（12-15小时训练）或 GPU（2-3小时）

4.文件架构 


5.数据集收集

数据集采用 multi30k 
可以使用以下方式进行下载

from torchnlp.datasets import multi30k_dataset
train = multi30k_dataset(train=True, dev=False, test=False)
dev = multi30k_dataset(train=False, dev=True, test=False)

数据集预处理：取一些常见的词，提升翻译质量，以构建词表，进行小写化，按空格分词，对序列长度进行限制

6.模型架构

基于源github项目中transformer模型 编码器-解码器

嵌入层：d_model=256，乘√d_model缩放
位置编码：
使用nn.Embedding将token id映射为d_model维向量
嵌入向量乘以 √d_model（缩放因子），使嵌入值的量级与后续位置编码相加时更为稳定。

采用论文中的正弦/余弦函数生成固定位置编码：
(pos,2i)=sin( 0000 2i/d modelpos ),PE (pos,2i+1) =cos( 10000 2i/d modelpos )

编码器：包含3个相同层堆叠而成，每层包含两个子层，分别是多头自注意力和前馈网络。

多头自注意力：
输入：(batch, src_len, d_model)
分别通过 W_Q, W_K, W_V 线性变换得到 Q, K, V，维度均为 (batch, src_len, d_model)。
将最后一维拆分为 nhead 个头，每个头维度 d_k = d_model / nhead = 32。
根据原文中的数学公式

\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V

拼接所有头的输出，再经 W_O 线性映射回 d_model。

掩码：仅使用 Padding Mask，防止注意力作用于 <pad> 位置。

前馈网络：d_ff=512,GELU激活，
结构：Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)
对序列中每个位置独立作用，参数共享。
解码器：
解码器由3个相同层堆叠，每层包含3个子层

掩码多头自注意力（Masked Self-Attention）
与编码器自注意力相同，但额外施加 后续掩码（Subsequent Mask） 和 Padding Mask，防止解码器看到当前时刻之后的信息。后续掩码是一个下三角矩阵，形状 (tgt_len, tgt_len)。

编码器-解码器交叉注意力（Cross-Attention）
Query 来自解码器上一子层的输出，Key 和 Value 来自编码器的最终输出。
掩码使用源序列的 Padding Mask，忽略源端的填充位置。

前馈网络：与编码器完全相同

输出层
解码器最后一层的输出通过一个线性层映射到目标语言词表大小（~12000），得到每个位置的 logits：

注意力掩码机制

Padding Mask
形状：(batch, 1, 1, seq_len)，可广播至注意力分数矩阵 (batch, nhead, seq_len_q, seq_len_k)。
值为 True 的位置表示有效 token，值为 False 表示 <pad>。
在注意力分数中，将无效位置设为 -1e9，使得 softmax 后的权重趋近于 0。

Subsequent Mask（解码器自注意力专用）
形状：(1, 1, tgt_len, tgt_len)，下三角为 True，上三角为 False。
与 Padding Mask 进行逻辑与运算，得到最终的 tgt_mask。

7.训练过程
python 数据集翻译.py
进入交互模式后，逐行输入英文句子，程序会输出对应的德文翻译。默认使用 Beam Search（宽度=3），翻译质量更高。

python 数据集翻译.py --text "a dog runs fast."（单句翻译）
python 数据集翻译.py --greedy --text "hello world."
python 数据集翻译.py --model best_model_standard.pth --text "a man is reading a book."

8.实验结果
基于默认配置（15,000 条训练数据，30 轮，d_model=256, layers=3）训练得到的模型表现如下：

指标	数值
最佳验证损失	~2.15（第 22 轮左右）
最终训练损失	~1.02
词表大小	源语言 11,892 / 目标语言 12,047
总参数量	12,482,304（约 12.5M）
训练总耗时（CPU, 8线程）	约 14 小时
推理速度（贪心解码）	~150ms/句（CPU）

示例翻译输出
英文输入	德文翻译（Beam Search）
a man in an orange hat starring at something.	ein mann in einem orangefarbenen hut starrt auf etwas.
a dog runs fast.	ein hund rennt schnell.
the cat sits on the mat.	die katze sitzt auf der matte.
a woman is reading a book.	eine frau liest ein buch.
children are playing football.	kinder spielen fußball.
a young girl is walking her dog in the park.	ein junges mädchen geht mit ihrem hund im park spazieren.
two men are sitting at a table eating lunch.	zwei männer sitzen an einem tisch und essen mittag.

训练曲线：
训练过程中记录的训练损失和验证损失曲线如下（自动生成 enhanced_training_curves.png）：
训练损失从约 4.13 快速下降，20 轮后稳定在 1.0 左右
验证损失在第 21轮达到最低（约 2.11），随后略有上升（轻微过拟合）
学习率预热阶段（前 3 轮）线性增长，之后余弦退火周期性衰减

9.模型参数统计
使用参数分析脚本可以获得详细的模型参数报告
python 模型参数分析.py

10.参考项目和参考文献

https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train_multi30k_de_en.sh
https://github.com/aladdinpersson/Machine-Learning-Collection
A Structured Self-attentive Sentence Embedding(Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio)
Attention is All You Need" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

