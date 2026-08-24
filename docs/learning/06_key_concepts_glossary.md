# 06 — Key Concepts Glossary: Baligh-1.7B v0

## Arabic NLP

| Term | Arabic | Definition |
|------|--------|------------|
| Fusha | الفصحى | Modern Standard Arabic (MSA). The formal written standard used in news, literature, and education. Baligh is trained to produce fusha. |
| Tashkeel | التشكيل | Diacritical marks (harakat) on Arabic letters: fatha, damma, kasra, sukun. Often stripped during normalization to reduce vocabulary. |
| Tatweel | ـ | Elongation character (U+0640). Visually stretches the preceding letter. Always removed during preprocessing. |
| Alef variants | | Alef can appear as plain (ا), with hamza above (أ), below (إ), or with madda (آ). Normalization maps all to plain alef. |
| Yaa variants | | Standard yaa (ي) vs tail yaa (ى). Normalization maps tail yaa to standard. |
| MinHash LSH | | Locality-Sensitive Hashing with MinHash signatures. Used for approximate near-duplicate detection on large Arabic text corpora. |

## Islamic Knowledge

| Term | Arabic | Definition |
|------|--------|------------|
| Fiqh | فقه | Islamic jurisprudence. The body of Islamic law derived from Quran, hadith, and scholarly consensus. |
| Tafsir | تفسير | Quranic exegesis. Interpretation and explanation of Quranic verses. |
| Hadith | حديث | Reports of the Prophet Muhammad's sayings, actions, and approvals. Second source of Islamic law after the Quran. |
| Aqidah | عقيدة | Islamic creed/theology. Core beliefs about God, prophets, angels, scriptures, and the hereafter. |
| Fatwa | فتوى | A legal opinion issued by a qualified Islamic scholar. Baligh is explicitly NOT a fatwa authority. |

## ML Training Concepts

| Term | Definition |
|------|------------|
| CPT (Continued Pretraining) | Domain adaptation stage. Training the base model on large amounts of domain-specific raw text (Arabic web data, Islamic texts) to improve language understanding before instruction tuning. |
| SFT (Supervised Fine-Tuning) | Instruction tuning stage. Training on (instruction, response) pairs to teach the model to follow instructions and produce helpful responses. |
| QLoRA | Quantized Low-Rank Adaptation. Combines 4-bit quantization of base weights with trainable low-rank adapter matrices. Reduces VRAM from ~12GB to ~4GB for a 1.5B model. |
| NF4 | Normal Float 4. A 4-bit quantization data type optimized for normally-distributed weights. Better than standard FP4 for neural network weights. |
| Double Quantization | Quantizing the quantization constants themselves. Saves an additional ~0.4 GB per billion parameters with negligible quality loss. |
| LoRA Rank (r) | The dimension of the low-rank adapter matrices. Higher rank = more expressiveness but more parameters. Baligh uses r=16. |
| LoRA Alpha | Scaling factor for adapter outputs. The effective scaling is alpha/r. With alpha=16, r=16, the scaling is 1.0 (no scaling). |
| Target Modules | Which layers get LoRA adapters. Baligh targets all 7: q_proj, k_proj, v_proj, o_proj (attention) + gate_proj, up_proj, down_proj (MLP). |
| Packing | Concatenating multiple short sequences into a single sample up to max_seq_length. Eliminates padding waste. Used in CPT, not SFT (to preserve conversation structure). |
| Response-Only Loss | Computing cross-entropy loss only on assistant tokens, masking user tokens. Implemented by TRL's SFTTrainer when using chat templates. |
| Gradient Checkpointing | Trading compute for memory by recomputing activations during backward pass instead of storing them. Reduces activation memory by ~70%. |
| Flash Attention 2 | Memory-efficient attention implementation that tiles the attention computation. Reduces memory from O(N^2) to O(N). |

## Hugging Face Ecosystem

| Term | Definition |
|------|------------|
| PEFT | Parameter-Efficient Fine-Tuning library. Provides LoRA, QLoRA, and other adapter methods. |
| TRL | Transformer Reinforcement Learning library. Provides SFTTrainer, DPOTrainer, etc. |
| BitsAndBytes | Library for quantized model loading (4-bit, 8-bit). Used with PEFT for QLoRA. |
| Dataset interleave | HF datasets function that alternates between multiple datasets by probability. |
| Arrow format | HF datasets' on-disk format. Memory-mapped, zero-copy loading for fast iteration. |
| Streaming mode | Loading datasets lazily from HF Hub without downloading the full dataset. Essential for 28B+ token datasets. |

## Quantization Formats

| Format | Use Case | Tool |
|--------|----------|------|
| GGUF | llama.cpp, Ollama, LM Studio | llama-cpp-python convert script |
| AWQ | vLLM, TensorRT-LLM | autoawq library |
| GPTQ | text-generation-inference | auto-gptq library |

## Benchmarks

| Benchmark | Type | What It Measures |
|-----------|------|------------------|
| MMLU-Arabic | Multiple choice | General knowledge across 57 subjects in Arabic |
| CIDAR | Generation | Arabic instruction-following quality (ROUGE, BLEU) |
| mr-tydi-arabic | Retrieval QA | Arabic passage retrieval and question answering |
| Islamic QA | Domain QA | Fiqh, hadith, tafsir knowledge accuracy |
| Perplexity | LM quality | How well the model predicts held-out Arabic text |
