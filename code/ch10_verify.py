import sys, torch, transformers; sys.path.insert(0, "code")
from calc import params, GPT2_SMALL, QWEN3_06B
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM
g = GPT2Config()
with torch.device("meta"): m = GPT2LMHeadModel(g)
n_g = sum(p.numel() for p in m.parameters())
q = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True,
    rope_theta=1_000_000, max_position_embeddings=40960)
with torch.device("meta"): mq = Qwen3ForCausalLM(q)
n_q = sum(p.numel() for p in mq.parameters())
print(n_g, params(GPT2_SMALL)["total"], n_q, params(QWEN3_06B)["total"], params(QWEN3_06B)["non_embedding"])
