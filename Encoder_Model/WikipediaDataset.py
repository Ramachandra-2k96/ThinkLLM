from datasets import load_dataset
from torch.utils.data import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer, chunk_size=512, chunk_overlap=24, subset_size=None):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Load dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        if subset_size is not None:
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        # Split all texts and flatten the list
        self.chunks: List[str] = []
        for text in dataset["text"]:
            text_chunks = self.text_splitter.split_text(text)
            self.chunks.extend(text_chunks)
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx) -> Dict:
        # Tokenize the chunk
        encodings = self.tokenizer(
            self.chunks[idx],
            truncation=True,
            padding='max_length',
            max_length=self.chunk_size,
            return_tensors="pt"
        )
        return {key: value.squeeze(0) for key, value in encodings.items()}