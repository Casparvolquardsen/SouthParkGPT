import logging
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import wandb
from utility import Config

LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


########## Based on nanoGPT from Andrej Karpathy ##########
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.num_heads
        self.n_embd = config.d_model
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logger.warning(
                "Using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.context_len, config.context_len)).view(
                    1, 1, config.context_len, config.context_len
                ),
            )

    def forward(self, x, attn_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.context_len, config.d_model)
                if config.positional_encoding == 'learned'
                else SinusoidalPositionalEmbedding(config),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
                ln_f=LayerNorm(config.d_model, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

        # report number of parameters
        logger.info("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.positional_encoding == 'learned':
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, attn_mask=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.context_len
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.context_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.context_len
        self.config.context_len = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]


# Generates a causal attention mask of the shape (size, size).
def generate_look_ahead_mask(size):
    # old: mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    # new from: https://discuss.pytorch.org/t/attn-mask-in-nn-multiheadattention/173603/3
    # seems like we got the attention mask wrong, it should be -inf for the future tokens
    # so we need to set the diagonal to 0 and the upper triangle to -inf
    arr = [[1 for _ in range(size)] for _ in range(size)]
    arr = torch.tensor(arr)
    mask = torch.triu(arr, diagonal=1)
    mask = mask == 1
    return mask


# Positional Encoding
# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)

        position = torch.arange(config.context_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model)
        )
        pe = torch.zeros(config.context_len, 1, config.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe = pe.permute(1, 0, 2)  # [1, max_len, d_model] since batch_first=True

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        position = torch.arange(config.context_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model)
        )
        pe = torch.zeros(config.context_len, config.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len]``
        """
        return self.pe[x]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.context_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        T = x.size(1)

        pos_embedding = self.position_embeddings(
            torch.arange(T, device=x.device)
        )  # shape (T, C)
        pos_embedding = self.dropout(pos_embedding)
        x = (
            x + pos_embedding
        )  # (T, C)+(T, C) broadcast the positional encoding to the batch size (B, T, C)
        return x


########## Baseline Models ##########
class RNNBaseline(nn.Module):
    def __init__(self, config: Config):
        super(RNNBaseline, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rnn = nn.LSTM(
            config.d_model,
            config.d_model,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.fc = FeedForward(n_embd=config.d_model, vocab_dim=config.vocab_size)

        try:
            wandb.watch(self, log_freq=100, log="all", log_graph=True)
        except:  # noqa: E722
            pass

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output)  # get all timesteps, i.e. processed sequences
        return output


class FeedForward(nn.Module):
    """the feed forward network (FFN) in the paper"""

    def __init__(self, n_embd, vocab_dim, dropout=0.2):
        super().__init__()
        # the paper (section 3.3) we have d_model=512 and d_ff=2048.
        # Therefore the inner layer is 4 times the size of the embedding layer
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, vocab_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
