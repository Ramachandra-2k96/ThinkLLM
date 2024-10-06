from WikipediaDataset import WikipediaDataset
from transformers import AutoTokenizer
from EnhancedCustomModelV2 import EnhancedCustomModelV2
from EnhancedCustomConfigV2 import EnhancedCustomConfigV2
from others import train_enhanced_custom_model_v2,save_enhanced_custom_model_v2
import traceback

def main():
    config = EnhancedCustomConfigV2(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        num_tree_layers=2,
        num_thought_steps=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        use_rezero=True,
        use_adapter=True,
        adapter_size=32,
        use_gated_ffn=True,
        use_moe=True,
        num_experts=4,
        top_k_experts=4,
        use_sparse_attention=True,
        sparse_attention_window=256,
        use_dynamic_ntk=True,
        ntk_alpha=0.5,
        use_mixture_of_experts=False,
        num_moe_experts=4,
        moe_top_k=4
    )

    # Update training configuration
    config.batch_size = 4
    config.epochs = 4
    config.learning_rate = 1e-4
    config.accumulation_steps = 4
    config.warmup_steps = 100

    model = EnhancedCustomModelV2(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)

    # Start with a very small dataset for testing
    dataset = WikipediaDataset(tokenizer, max_length=512, subset_size=1000)

    try:
        model = train_enhanced_custom_model_v2(model, dataset, config)
        save_enhanced_custom_model_v2(model, tokenizer, "enhanced_custom_model_v2_final")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()