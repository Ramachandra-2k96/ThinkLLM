from torch.optim import AdamW
import os
from transformers import PreTrainedModel, PretrainedConfig
import torch
from DecoderOnlyConfig import DecoderOnlyConfig
from ImprovedDecoderOnlyGPT import ImprovedDecoderOnlyGPT
from transformers import GPT2Tokenizer
from transformers import get_cosine_schedule_with_warmup

def save_enhanced_model(model, optimizer, scheduler, tokenizer, config, epoch, train_loss_history, path):
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # 1. Save the model state
        model_to_save = model.module if hasattr(model, 'module') else model
        if isinstance(model_to_save, PreTrainedModel):
            model_to_save.save_pretrained(path, safe_serialization=False)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # 2. Save the tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        
        # 3. Save the config
        if config is not None:
            config.save_pretrained(path)
        
        # 4. Save the training state
        training_state = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss_history': train_loss_history,
        }
        
        # 5. Save additional components
        additional_components = {}
        for i, block in enumerate(model.h):
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'experts'):
                additional_components.setdefault('moe_states', {})[i] = block.mlp.state_dict()
            if hasattr(block, 'ntk'):
                additional_components.setdefault('ntk_states', {})[i] = block.ntk.state_dict()
            if hasattr(block, 'decision_trees'):
                additional_components.setdefault('decision_tree_states', {})[i] = block.decision_trees.state_dict()
            if hasattr(block, 'cognitive'):
                additional_components.setdefault('cognitive_states', {})[i] = block.cognitive.state_dict()
        
        # Combine all states
        full_state = {
            'training_state': training_state,
            'additional_components': additional_components
        }
        
        # Save the combined state
        torch.save(full_state, os.path.join(path, 'training_state.bin'))
        
        print(f"Model and training state successfully saved to {path}")
    except Exception as e:
        print(f"Error occurred while saving the model: {str(e)}")
        raise

def load_enhanced_model(path, training=False):
    # 1. Load config and model
    config = DecoderOnlyConfig.from_pretrained(path)
    model = ImprovedDecoderOnlyGPT.from_pretrained(path, config=config)
    
    # 2. Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    
    if not training:
        return model, tokenizer
    
    # 3. Load training state
    training_state_path = os.path.join(path, 'training_state.bin')
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(f"Training state not found at {training_state_path}")
    
    full_state = torch.load(training_state_path)
    training_state = full_state['training_state']
    additional_components = full_state['additional_components']
    
    # 4. Restore additional components
    for i, block in enumerate(model.h):
        if hasattr(block.mlp, 'experts'):
            block.mlp.load_state_dict(additional_components['moe_states'][i])
        if hasattr(block, 'ntk'):
            block.ntk.load_state_dict(additional_components['ntk_states'][i])
        if hasattr(block, 'decision_trees'):
            block.decision_trees.load_state_dict(additional_components['decision_tree_states'][i])
        if hasattr(block, 'cognitive'):
            block.cognitive.load_state_dict(additional_components['cognitive_states'][i])
    
    # 5. Create and load optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    
    # 6. Create and load scheduler if it exists
    scheduler = None
    if training_state['scheduler_state_dict'] is not None:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_training_steps
        )
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
    
    # 7. Restore memory states if they exist
    if 'memory_states' in training_state:
        for i, block in enumerate(model.h):
            if hasattr(block, 'cognitive'):
                block.cognitive.memory_state = training_state['memory_states'][i]
    
    return model, tokenizer, optimizer, scheduler, training_state['epoch'], training_state['train_loss_history']