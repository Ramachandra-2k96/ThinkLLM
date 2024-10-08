from transformers import PretrainedConfig

class EnhancedCustomConfigV2(PretrainedConfig):
    model_type = "enhanced_custom_v2"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_tree_layers=4,
        num_thought_steps=4,
        tie_word_embeddings=True,
        use_rezero=True,
        use_adapter=True,
        adapter_size=64,
        use_gated_ffn=True,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_sparse_attention=True,
        sparse_attention_window=256,
        use_dynamic_ntk=True,
        ntk_alpha=0.5,
        use_mixture_of_experts=True,
        num_moe_experts=8,
        moe_top_k=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_tree_layers = num_tree_layers
        self.num_thought_steps = num_thought_steps
        self.use_rezero = use_rezero
        self.use_adapter = use_adapter
        self.adapter_size = adapter_size
        self.use_gated_ffn = use_gated_ffn
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_window = sparse_attention_window
        self.use_dynamic_ntk = use_dynamic_ntk
        self.ntk_alpha = ntk_alpha
        self.use_mixture_of_experts = use_mixture_of_experts
        self.num_moe_experts = num_moe_experts
        self.moe_top_k = moe_top_k