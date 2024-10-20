from transformers import GPT2Tokenizer
from save_load  import load_enhanced_model,save_enhanced_model
from DecoderOnlyConfig import DecoderOnlyConfig
from ImprovedDecoderOnlyGPT import ImprovedDecoderOnlyGPT
from WikipediaDataset import WikipediaDataset
from Train import train_chatbot_model

def main(num_examples=None, resume_from=None):
    # Training configuration
    training_config = {
        'batch_size': 2,
        'num_epochs': 3,
        'learning_rate': 1e-4,
        'warmup_steps': 100,
        'max_length': 512,
    }
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize or load model and training state
    if resume_from:
        print(f"Resuming training from {resume_from}")
        model, tokenizer, optimizer, scheduler, start_epoch, train_loss_history = load_enhanced_model(
            resume_from, training=True
        )
    else:
        print("Initializing new model")
        config = DecoderOnlyConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=512,
            n_embd=512,
            n_layer=6,
            n_head=8,
            n_inner=1536,
            learning_rate=training_config['learning_rate'], 
            gradient_accumulation_steps=8,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            use_moe=True,
            num_experts=2,
            top_k_experts=2,
            use_cognitive_layer=True,
            use_ntk_layer=True,
            use_decision_trees=True,
            num_epochs=training_config['num_epochs'],
            batch_size=training_config['batch_size'],
            warmup_steps=training_config['warmup_steps'],
        )
        model = ImprovedDecoderOnlyGPT(config)
        start_epoch = 0
        train_loss_history = None

    # Create dataset
    dataset = WikipediaDataset(
        tokenizer, 
        max_length=config.n_positions,
        num_examples=num_examples
    )

    # Train the model
    model, train_loss_history, scheduler, optimizer  = train_chatbot_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        start_epoch=start_epoch,
        train_loss_history=train_loss_history
    )


    # Save the final model
    save_enhanced_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            config=config,
            epoch=config.num_epochs,
            train_loss_history=train_loss_history,
            path="final_enhanced_wikipedia_chatbot"
    )

    return model, tokenizer, train_loss_history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train enhanced chatbot model')
    parser.add_argument('--num_examples', type=int, default=None, 
                        help='Number of examples to use for training (default: all)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    trained_model, tokenizer, loss_history = main(
        num_examples=args.num_examples,
        resume_from=args.resume_from
    )