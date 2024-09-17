from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from .utils import load_model, prepare_data
from datasets import load_dataset

def evaluate_model(model_path: str, dataset_name: str):
    # Model ve tokenizer'ı yükle
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Veri kümesini Hugging Face'ten yükle
    dataset = load_dataset("takala/financial_phrasebank", split="test")  # Veri seti adı güncellendi
    tokenized_datasets = prepare_data(dataset, tokenizer)

    # Trainer ve değerlendirme
    trainer = Trainer(model=model)
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets)
    print(f"Değerlendirme Sonuçları: {eval_results}")

if __name__ == "__main__":
    evaluate_model(model_path="./models/fine-tuned-llama", dataset_name="takala/financial_phrasebank")
