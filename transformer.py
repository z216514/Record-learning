import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


#基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs) 
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens) #第一层线性变换，将输入从 ffn_num_input 维映射到 ffn_num_hiddens 维。
        self.relu = nn.ReLU() #ReLU激活函数，用于引入非线性。
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs) #第二层线性变换，将 ffn_num_hiddens 维的输出映射回 ffn_num_outputs 维。

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
#输入 X 首先通过 self.dense1 进行线性变换。
#然后应用 self.relu 激活函数。
#最后通过 self.dense2 进行第二次线性变换，得到最终输出。


ffn = PositionWiseFFN(4, 4, 8) #创建了一个前馈网络，输入和隐藏层的大小都是 4，输出层的大小是 8。
ffn.eval() #调用 .eval() 方法将模型设置为评估模式
ffn(torch.ones((2, 3, 4)))[0]  #创建了一个形状为 (2, 3, 4) 并为1的张量，然后，你将这个张量传递给 ffn 进行前向传播。访问输出张量的第一个元素



#残差连接和层规范化
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2) #创建了两个不同的归一化层：LayerNorm 和 BatchNorm1d
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32) #创建了一个形状为 (2, 2) 的张量，其中包含两个样本，每个样本有两个特征。
# 在训练模式下计算X的均值和方差
print('layer norm:', ln(X), '\nbatch norm:', bn(X))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) #一个 nn.Dropout 层，根据提供的 dropout 比率丢弃一部分输入。
        self.ln = nn.LayerNorm(normalized_shape) #对输入进行规范化，使得每个特征的均值为0，方差为1。

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape


#编码器
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):   #分别代表多头注意力机制中键（Key）、查询（Query）、值（Value）的维度。Dropout比率，用于正则化。
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias) #一个 d2l.MultiHeadAttention 层，实现多头注意力机制。
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)  #两个 AddNorm 层，分别用于注意力输出和前馈网络输出的归一化和残差连接。

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
""""
输入 X 首先通过 self.addnorm1，其中 self.attention 计算注意力输出。
然后，X 与注意力输出相加，并通过 self.addnorm1 进行归一化。
接着，结果通过 self.ffn 前馈网络。
最后，前馈网络的输出再次通过 self.addnorm2 进行归一化。
""""



X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5) #创建了一个 EncoderBlock 实例
encoder_blk.eval()
encoder_blk(X, valid_lens).shape


class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens) #一个 nn.Embedding 层，用于将输入的索引转换为对应的嵌入向量。
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout) #一个 d2l.PositionalEncoding 层，用于给嵌入向量添加位置信息。
        self.blks = nn.Sequential() #用于存储多个 EncoderBlock。
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
"""
输入 X 首先通过嵌入层和位置编码层。
嵌入向量乘以嵌入维度的平方根进行缩放，以防止位置编码的值过大。
然后，输入通过多个编码器块（self.blks），每个块都包含多头注意力和前馈网络。
在前向传播过程中，还存储了每个编码器块的注意力权重。
"""


encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape
#指定了超参数来创建一个两层的Transformer编码器。 Transformer编码器输出的形状是（批量大小，时间步数目，num_hiddens）。




#解码器
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout) #第一个 d2l.MultiHeadAttention 层，用于自注意力机制。
        self.addnorm1 = AddNorm(norm_shape, dropout) #第一个 AddNorm 层，用于自注意力输出的归一化和残差连接。
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout) #第二个 d2l.MultiHeadAttention 层，用于编码器-解码器注意力机制。
        self.addnorm2 = AddNorm(norm_shape, dropout) #第二个 AddNorm 层，用于编码器-解码器注意力输出的归一化和残差连接。
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens) #PositionWiseFFN 层，实现前馈网络。
        self.addnorm3 = AddNorm(norm_shape, dropout) #第三个 AddNorm 层，用于前馈网络输出的归一化和残差连接。

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

"""
输入 X 代表当前步骤的输入，state 是一个包含编码器输出和有效长度信息的状态元组。
state[0] 包含编码器的输出 enc_outputs。
state[1] 包含编码器的有效长度 enc_valid_lens。
state[2] 包含解码器的中间状态，用于存储每个块的解码输出。
根据是否处于训练模式，生成解码器的有效长度 dec_valid_lens。
首先进行自注意力，然后是编码器-解码器注意力，最后是前馈网络。
在每一步，输出都通过相应的 AddNorm 层进行归一化和残差连接。
最终返回前馈网络的输出和更新后的状态。
"""

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape



class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]  
#接受编码器的输出 enc_outputs 和有效长度 enc_valid_lens
#并返回解码器的初始状态，包括编码器的输出、有效长度以及一个用于存储每层解码器状态的列表。

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights




#训练
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10  #定义了模型和训练过程中使用的超参数
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)                                                                     #实例化 TransformerEncoder 类，使用之前定义的超参数和模型组件维度。
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)                                                                     #实例化 TransformerDecoder 类，同样使用之前定义的参数。
net = d2l.EncoderDecoder(encoder, decoder)                                    #使用 d2l.EncoderDecoder 将编码器和解码器组合成一个完整的seq2seq模型。
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)




engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')  #翻译一组英文句子 engs 到法文，并打印出翻译结果和相应的BLEU分数。
