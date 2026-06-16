# Key Concepts Glossary - Baligh-1.5B v0

## Arabic NLP Terms

| Term | Arabic | Definition |
|------|--------|------------|
| **Fusha (الفصحى)** | Modern Standard Arabic | Standardized, literary Arabic used in media, education, formal writing across Arab world |
| **Tashkeel (التشكيل)** | Diacritical marks (harakat) | Short vowel marks (fatha, damma, kasra, sukun) placed on consonants; often stripped in NLP |
| **Tatweel (ـ)** | Elongation character | Visual stretching of Arabic words; no semantic meaning; removed during normalization |
| **Arabic Normalization** | التجويد العربي | Canonicalizing character variants: أ/إ/آ → ا, ى → ي, removal of ZWNJ/ZWJ |
| **MinHash LSH** | التجزئة المحلية المتجهة | Locality-Sensitive Hashing with MinHash for approximate near-duplicate detection |
| **Shingle (k-gram)** | النطاق | Contiguous sequence of k items (words/chars); used for Jaccard similarity estimation |

---

## Islamic Knowledge Terms

| Term | Arabic | Definition |
|------|--------|------------|
| **Fiqh (فقه)** | Islamic jurisprudence | Human understanding/application of Sharia; derived from Quran, Hadith, consensus, analogy |
| **Tafsir (تفسير)** | Quranic exegesis | Interpretation/explanation of Quranic verses; linguistic, legal, theological |
| **Hadith (حديث)** | Reports of Prophet's sayings/actions | Secondary source of Islamic law; graded by authenticity (sahih, hasan, da'if) |
| **Aqidah (عقيدة)** | Islamic creed/theology | Core beliefs: Tawhid, angels, books, prophets, Day of Judgment, Qadar |
| **Fatwa (فتوى)** | Legal opinion | Non-binding legal ruling by qualified scholar (mufti); this model is NOT an authority |
| **Sharia (الشريعة)** | Islamic law | Divine law from Quran/Sunnah; Fiqh is human understanding of Sharia |
| **Sunnah (السنة)** | Prophetic tradition | Practices/habits of Prophet Muhammad; second source after Quran |
| **Sahih (صحيح)** | Authentic | Hadith grading: sound chain, reliable narrators, no defects |
| **Isnad (الإسناد)** | Chain of narrators | Chain of transmission for a Hadith; critical for authenticity grading |

---

## ML Training Terms

| Term | Definition |
|------|------------|
| **CPT (Continued Pretraining)** | Domain adaptation on raw text corpora; model learns language patterns, vocabulary, domain knowledge |
| **SFT (Supervised Fine-Tuning)** | Instruction tuning on (prompt, response) pairs; teaches instruction following, style, formatting |
| **QLoRA (Quantized LoRA)** | 4-bit quantized base model + Low-Rank Adapters; freezes base weights, trains ~1-2% params |
| **LoRA (Low-Rank Adaptation)** | Decomposes weight update ΔW = A×B where A∈R^(d×r), B∈R^(r×d); r << d |
| **Unsloth** | Training toolkit with fused kernels for 2x faster QLoRA; optimized Triton kernels |
| **NF4 (NormalFloat4)** | 4-bit quantization optimized for normally distributed weights; better than uniform INT4 |
| **Double Quantization** | Quantizes quantization constants (scales); saves additional 0.4 bits/parameter |
| **Response-only Loss** | Loss computed only on assistant tokens; user tokens masked with -100 |
| **Packing** | Concatenating multiple sequences to fill `max_seq_length`; improves GPU utilization |
| **Gradient Checkpointing** | Recompute activations during backward pass; trades compute for memory |
| **Gradient Accumulation** | Accumulate gradients over N steps before optimizer step; simulates larger batch |
| **Flash Attention 2** | Memory-efficient attention kernel; O(N²) → O(N) memory via tiling |
| **AdamW Fused** | Fused AdamW kernel (single kernel for step); 20-30% faster |
| **bfloat16** | Brain Float 16; same exponent range as FP32, 7-bit mantissa; no gradient scaling needed |

---

## Training Hyperparameters

| Term | Baligh Value | Explanation |
|------|--------------|-------------|
| **LoRA Rank (r)** | 16 | Rank of adaptation matrices; 16 = 16×d + d×16 params per layer |
| **LoRA Alpha (α)** | 16 | Scaling factor; effective scale = α/r = 1.0 |
| **LoRA Dropout** | 0.0 | No dropout on LoRA; base model frozen provides regularization |
| **Target Modules** | All attention + MLP | q/k/v/o_proj + gate/up/down_proj |
| **Learning Rate (CPT)** | 2e-4 | Standard for continued pretraining |
| **Learning Rate (SFT)** | 1e-4 | Lower for alignment; prevents catastrophic forgetting |
| **Warmup Steps** | 1000 (CPT) / 500 (SFT) | Linear warmup; prevents early instability |
| **LR Scheduler** | Cosine | Smooth decay to 0; better final convergence |
| **Weight Decay** | 0.01 | L2 regularization on weights |
| **Max Grad Norm** | 1.0 | Gradient clipping; prevents explosion |
| **Batch Size** | 2 (per device) | Fits in 8-12GB VRAM with QLoRA |
| **Grad Accum** | 4 | Effective batch = 8 |
| **Max Seq Length** | 2048 | Training context; extendable to 32K |
| **Mixed Precision** | bf16 | Bfloat16; no gradient scaling needed |

---

## Model Architecture (Qwen2.5-1.5B)

| Component | Specification |
|-----------|---------------|
| **Parameters** | 1.54B |
| **Layers** | 28 |
| **Hidden Size** | 2048 |
| **Attention Heads** | 16 (Grouped Query Attention - GQA) |
| **KV Heads** | 8 (GQA ratio 2:1) |
| **Hidden Act** | SwiGLU (MLP) |
| **Max Context** | 32,768 tokens |
| **Vocab Size** | 151,936 |
| **Tokenizer** | BPE (Unigram); Arabic supported natively |

---

## Quantization Formats

| Format | Use Case | Quality | Speed | Compatibility |
|--------|----------|---------|-------|---------------|
| **GGUF q4_k_m** | CPU/edge (llama.cpp) | ★★★★☆ | Fast | Universal (llama.cpp, Ollama, LM Studio) |
| **AWQ 4-bit** | GPU inference | ★★★★★ | Fastest | vLLM, TGI, TensorRT-LLM |
| **GPTQ 4-bit** | GPU inference | ★★★★☆ | Fast | AutoGPTQ, ExLlamaV2 |
| **FP16/BF16** | Training/Full precision | ★★★★★ | Baseline | Native PyTorch |

**Recommendation**: GGUF q4_k_m for distribution; AWQ for production GPU serving

---

## Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Perplexity (PPL)** | exp(avg cross-entropy loss) | Language modeling quality |
| **ROUGE-1/2/L** | N-gram overlap (recall-oriented) | Summarization, QA |
| **BLEU** | N-gram precision + brevity penalty | Translation, generation |
| **BERTScore** | Contextual embedding similarity | Semantic equivalence |
| **Exact Match** | String equality (normalized) | MCQ, closed QA |
| **Accuracy** | Correct predictions / total | MCQ benchmarks (MMLU) |
