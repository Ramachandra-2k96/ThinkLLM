from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        if subset_size is not None:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        self.data = dataset["text"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.tokenizer(self.data[idx], max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")