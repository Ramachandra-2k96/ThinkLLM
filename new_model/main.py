from WikipediaDataset import WikipediaDataset
from transformers import AutoTokenizer
from EnhancedCustomModelV2 import EnhancedCustomModelV2
from EnhancedCustomConfigV2 import EnhancedCustomConfigV2
from others import train_enhanced_custom_model_v2,save_enhanced_custom_model_v2


def main():
    config = EnhancedCustomConfigV2(
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_tree_layers=4,
        num_thought_steps=4,
        max_position_embeddings=512,
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
        moe_top_k=2
    )

    config.batch_size = 4
    config.epochs = 3
    config.learning_rate = 1e-3
    config.accumulation_steps = 1000
    config.warmup_steps = 1000

    model = EnhancedCustomModelV2(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)

    dataset = WikipediaDataset(tokenizer, max_length=512, subset_size=500)

    train_enhanced_custom_model_v2(model, dataset, config)

    save_enhanced_custom_model_v2(model, tokenizer, "enhanced_custom_model_v2_final")

if __name__ == "__main__":
    main()