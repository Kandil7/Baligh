"""10-step smoke test for Baligh-1.7B CPT+SFT pipeline on RTX 5000."""

import json
import os
import tempfile
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "unsloth/Qwen3-1.7B-Base"
STEPS = 10
MAX_SEQ_LEN = 512


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Free VRAM: {free_gb:.1f} GB")

    # --- Load tokenizer ---
    print(f"\n[1/5] Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab: {len(tokenizer)}, Pad: {tokenizer.pad_token}")

    # --- Load model ---
    print(f"\n[2/5] Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    free_after = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  Loaded. Free VRAM after load: {free_after:.1f} GB")

    # --- Create tiny dummy dataset ---
    print(f"\n[3/5] Creating dummy CPT dataset ({STEPS} examples)")
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
    print(f"  Dataset: {len(dataset)} examples, seq_len={MAX_SEQ_LEN}")

    # --- Run short CPT training ---
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[4/5] Running CPT ({STEPS} steps) -> {tmpdir}")
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            bf16=True,
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
        loss = result.metrics.get("train_loss", "N/A")
        print(f"  CPT done. Final loss: {loss}")

    # --- Quick inference test ---
    print(f"\n[5/5] Inference test")
    prompt = "بسم الله"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Input:  {prompt}")
    print(f"  Output: {response[:200]}")

    # --- VRAM summary ---
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    current_mb = torch.cuda.memory_allocated() / 1e6
    print(f"\n--- VRAM Summary ---")
    print(f"  Peak allocated: {peak_mb:.0f} MB")
    print(f"  Current: {current_mb:.0f} MB")
    print(f"\n✅ Smoke test PASSED — Baligh-1.7B pipeline works on RTX 5000.")


if __name__ == "__main__":
    main()
