"""10-step smoke test for Baligh-1.7B CPT+SFT pipeline on RTX 5000."""

import os
import tempfile

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "unsloth/Qwen3-1.7B-Base"
STEPS = 10
MAX_SEQ_LEN = 256


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    torch.cuda.mem_get_info()[0] / 1e9

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    torch.cuda.mem_get_info()[0] / 1e9

    # --- Create tiny dummy dataset ---
    arabic_texts = [
        "بسم الله الرحمن الرحيم. الحمد لله رب العالمين.",
        "قال النبي صلى الله عليه وسلم: إنما الأعمال بالنيات.",
        "العلم نور والجهل ظلام. طلب العلم فريضة على كل مسلم.",
        "القرآن الكريم هو كتاب الله Анزمه على المسلمين Чтوة والتدبر.",
        "التاريخ الإسلامي حافل بالأحداث والإنجازات العظيمة.",
        "اللغة العربية من أقدم اللغات السامية وأكثرها انتشاراً.",
        "الصلاة هي الركن الثاني من أركان الإسلام الخمسة.",
        "الزكاة فريضة مالية تُخرج لمستحقيها سنوياً.",
        "الحج هو زيارة بيت الله الحرام لمن استطاع إليها طريقاً.",
        "الصيام في شهر رمضان واجب على كل مسلم بالغ عاقل.",
    ]
    dataset = Dataset.from_dict({"text": arabic_texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # --- Run short CPT training ---
    with tempfile.TemporaryDirectory() as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            fp16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=9999,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )

        from transformers import DataCollatorForLanguageModeling, Trainer

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        result = trainer.train()
        result.metrics.get("train_loss", "N/A")

    # --- Quick inference test ---
    prompt = "بسم الله"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- VRAM summary ---
    torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.memory_allocated() / 1e6


if __name__ == "__main__":
    main()
