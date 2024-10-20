from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2Tokenizer
import re
from tqdm.auto import tqdm

class WikipediaDataset(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 512, num_examples: int = None, min_length: int = 50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length

        # Load and preprocess Wikipedia dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:300]")
        
        # Text splitter with overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=50,  # Overlap to keep coherence
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!"]
        )
        
        self.texts = []
        for item in tqdm(dataset, desc="Processing Wikipedia articles"):
            cleaned_text = self._clean_text(item['text'])  # Preprocess each article
            chunks = text_splitter.split_text(cleaned_text)
            # Filter out very small chunks to avoid noise
            self.texts.extend([chunk for chunk in chunks if len(chunk) >= self.min_length])
        
        # Limit the number of examples if specified
        if num_examples is not None:
            self.texts = self.texts[:num_examples]

    def _clean_text(self, text: str) -> str:
        """Cleans Wikipedia text by removing unwanted parts."""
        text = re.sub(r'\[\d+\]', '', text)  # Remove reference brackets like [1], [2]
        text = re.sub(r'\(.*?\)', '', text)  # Remove text within parentheses (optional)
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text with truncation and padding
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Create labels (shifted input_ids for language modeling)
        labels = inputs['input_ids'].clone()
        labels[:, :-1] = inputs['input_ids'][:, 1:]  # Shift inputs to create labels
        labels[:, -1] = -100  # Ignore prediction of the last token (as it's impossible to predict)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }