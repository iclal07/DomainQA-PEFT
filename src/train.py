from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from .utils import prepare_data, save_model

def train_model(model_name: str, dataset_name: str, output_dir: str, num_epochs: int = 3, batch_size: int = 4):
    # Model ve tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

    # LoRA yapılandırmasını oluştur
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    
    # PEFT modelini oluştur
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Veri kümesini Hugging Face'ten yükle
    dataset = load_dataset("takala/financial_phrasebank", split="train")  # Veri seti adı güncellendi
    tokenized_datasets = prepare_data(dataset, tokenizer)

    # Eğitim parametrelerini ayarla
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
    )


