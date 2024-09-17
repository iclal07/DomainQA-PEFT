import torch
from datasets import DatasetDict
from transformers import AutoTokenizer

def prepare_data(dataset, tokenizer, max_length: int = 128) -> DatasetDict:
    # Tokenizasyon fonksiyonu
    def tokenize_function(examples):
        return tokenizer(examples["question"], padding="max_length", truncation=True, max_length=max_length)
    
    # Veriyi tokenle ve böl
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets.train_test_split(test_size=0.1)

def save_model(model, output_dir: str):
    # Model ve tokenizer'ı kaydet
    model.save_pretrained(output_dir)
    print(f"Model {output_dir} konumuna kaydedildi.")

def load_model(model_path: str):
    # Modeli yükle
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return model
