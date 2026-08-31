<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Baligh assets reviewed

I inspected both linked resources:

- **GitHub repository:** `Kandil7/Baligh` contains the project README, source code under `src/`, configuration files, Colab CLI utilities, Docker files, documentation, requirements, and tests.
- **Hugging Face model:** `Kandil7/Baligh-1.5B` is an Arabic text-generation model fine-tuned from `Qwen/Qwen2.5-1.5B-Instruct`, with tags indicating QLoRA, Unsloth, RAG, knowledge grounding, and Islamic AI. It uses the Apache-2.0 license and was updated on June 15, 2026.

The project structure appears production-oriented, while the model card metadata identifies Baligh as an Arabic, Islamic-domain fine-tune rather than a standalone foundation model. For Sunni alignment, the next important review should verify the actual training data, refusal policy, evaluation methodology, inference template, and whether generated religious answers are explicitly grounded in reliable Sunni sources rather than relying only on fine-tuning behavior.

## Artifact

No file was created.

## Next actions

1. Review the full README and training/configuration files to document the exact fine-tuning pipeline.
2. Audit the Hugging Face model card for training data, intended use, limitations, and inference examples.
3. Add Sunni-specific evaluations covering Qur’an and hadith attribution, madhhab-sensitive fiqh, creed, hallucination, and citation faithfulness.

---

# ايه افضل model اعمل عليه fine tuning ل Baligh-1.5B

لو هدفك تطوير **Baligh** كمساعد عربي سُنّي مع RAG، فأفضل اختيار عملي هو الاستمرار على **Qwen2.5-1.5B-Instruct** في النسخة الحالية، ثم تجربة **Qwen3-1.7B** كمسار ترقية منفصل—not استبدال مباشر. نسختك الحالية مبنية أصلًا على `Qwen/Qwen2.5-1.5B-Instruct` وموسومة بدعم العربية وQLoRA وUnsloth وRAG.

## الترشيح الأساسي

### 1. Qwen2.5-1.5B-Instruct — الأفضل لاستكمال Baligh

اختياري الأول حاليًا:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

الأسباب:

- يحافظ على توافق الـ tokenizer والـ chat template مع Baligh الحالي.
- يقلل خطر تراجع الأداء بسبب تغيير الـ base model.
- مناسب جدًا لـ QLoRA على Colab أو GPU متوسط.
- جيد لبناء سلوك متخصص: الالتزام بالمصادر، الإجابة بالعربية، الرفض عند غياب الدليل، وتنسيق إجابات RAG.
- أسهل في عمل مقارنة عادلة بين checkpoints القديمة والجديدة.

بالنسبة لمشروعك، **جودة البيانات والسياسة السلوكية أهم من الانتقال من 1.5B إلى نموذج آخر قريب الحجم**. لا أنصح بإعادة بدء المشروع على base مختلف قبل تثبيت benchmark واضح لنسخة Baligh الحالية.

## مسار الترقية

### 2. Qwen3-1.7B — أفضل تجربة ترقية

جرّب:

```text
Qwen/Qwen3-1.7B
```

كـ branch تجريبي مستقل، خصوصًا إذا كنت تريد تحسين:

- اتباع التعليمات.
- الاستدلال متعدد الخطوات.
- استخدام الأدوات وRAG.
- التبديل بين العربية والإنجليزية.
- جودة الإجابات في الأسئلة المركبة.

البحث الحالي يصف Qwen3 1.7B بأنه نموذج متعدد اللغات يدعم العربية، لكن يجب اختبار النسخة الرسمية والـ tokenizer والـ chat template عمليًا قبل اعتماده كأساس نهائي.[^2_1]

مهم: لا تعمل merge أو continuation training بين Qwen2.5 وQwen3. استخدم نفس dataset ونفس evaluation suite للمقارنة، ثم اختر الأفضل بالأرقام.

## اختيارات لا أنصح بها الآن

| النموذج | القرار | السبب |
| :-- | :-- | :-- |
| Qwen2.5-1.5B-Instruct | أساسي | توافق مباشر مع Baligh الحالي |
| Qwen3-1.7B | تجربة ترقية | احتمال تحسن reasoning وtool use، لكن يحتاج benchmark |
| Falcon-H1-1.5B-Instruct | تجربة ثانوية | قد يكون جيدًا عربيًا، لكن التوافق والمنظومة أقل وضوحًا لمشروعك |
| RightNow-Arabic-0.5B-Turbo | غير مناسب كـ base رئيسي | أصغر بكثير؛ يصلح edge أو baseline عربي |
| LFM2.5-1.2B-Instruct | تجربة لاحقة | promising، لكن تغيير architecture/ecosystem قد يصعّب نقل pipeline |
| Qwen3-ASR-1.7B | غير مناسب | هذا نموذج speech/ASR، وليس base مناسبًا لمساعد نصي |

تُظهر نتائج منشورة أن Falcon-H1-1.5B-Instruct منافس في بعض اختبارات العربية، بينما RightNow-Arabic-0.5B موجّه أكثر لنموذج عربي صغير متخصص؛ لكن هذه النتائج ليست كافية وحدها للحكم على الفقه، العقيدة، صحة النقل، أو الالتزام بالمصادر السنية.[^2_2]

## الخطة التي أوصي بها

### المرحلة الأولى: تثبيت Baligh الحالي

درّب على:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

باستخدام:

- QLoRA أو LoRA.
- `response_only_loss`.
- chat template الأصلي.
- packing بحذر مع فصل الأمثلة.
- نسبة بيانات عامة صغيرة للحفاظ على القدرات العامة.
- بيانات دينية سُنّية عالية الجودة بدل ضخ نصوص خام فقط.

تقسيم مبدئي للبيانات:


| النوع | النسبة المقترحة |
| :-- | --: |
| تعليمات عربية عامة | 20–30% |
| أسئلة وأجوبة إسلامية موثقة | 25–35% |
| أمثلة RAG مع مصادر | 20–30% |
| رفض وتصحيح الهلوسة | 10–15% |
| أسلوب وصياغة عربية | 5–10% |

النسب ليست قانونًا ثابتًا؛ الأهم أن تكون بيانات العقيدة والفتوى مرتبطة بمصادر واضحة، مع التفريق بين **النص المنقول** و**الشرح** و**الاستنباط** و**عدم العلم**.

### المرحلة الثانية: تدريب Qwen3-1.7B بنفس البيانات

أنشئ تجربة منفصلة:

```text
Baligh-Qwen3-1.7B
```

ولا تقارن loss فقط. قارن على مجموعة ثابتة تشمل:

- صحة نسبة الآيات والأحاديث.
- عدم اختلاق التخريج أو رقم الحديث.
- الالتزام بالمذهب أو التصريح بوجود خلاف معتبر.
- التعامل مع الأسئلة العقدية وفق منهج أهل السنة والجماعة.
- رفض إصدار فتوى شخصية عند نقص المعطيات.
- الالتزام بسياق RAG وعدم مخالفة المصدر بلا تعليل.
- جودة العربية الفصحى.
- مقاومة prompt injection داخل الوثائق المسترجعة.


## إعداد عملي مبدئي

لبداية آمنة على QLoRA:

```yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct
max_seq_length: 4096
load_in_4bit: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 1.0e-4
num_train_epochs: 2
warmup_ratio: 0.05
weight_decay: 0.01
packing: true
response_only_loss: true
```

ثم جرّب:

```yaml
model_name: Qwen/Qwen3-1.7B
max_seq_length: 4096
load_in_4bit: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 5.0e-5
num_train_epochs: 1-2
```

استخدم learning rate أقل في Qwen3 كبداية، لأنك لا تريد أن تمحو قدراته العامة أثناء إدخال السلوك الديني المتخصص.

## القرار النهائي

ترتيبي لمشروع Baligh:

1. **Qwen2.5-1.5B-Instruct**: الخيار الإنتاجي الأقرب والأكثر أمانًا الآن.
2. **Qwen3-1.7B**: أفضل تجربة ترقية يجب تنفيذها بbenchmark موحد.
3. **Falcon-H1-1.5B-Instruct**: baseline إضافي للمقارنة، وليس اختياري الأول.
4. **نموذج عربي 0.5B**: فقط إذا كان الهدف تشغيلًا شديد الخفة، وليس أفضل جودة دينية.

وبما أن Baligh الحالي منشور بترخيص Apache-2.0 وبوصف عربي/إسلامي/مُحسّن بـ QLoRA، فالأفضل إصدار نسخة جديدة باسم واضح مثل `Baligh-1.5B-v2` مع إبقاء النسخة الحالية baseline ثابتًا.

## Artifact

لا يوجد ملف مُنشأ.

## Next actions

1. درّب نفس dataset على Qwen2.5-1.5B وQwen3-1.7B بدون تغيير pipeline.
2. أنشئ benchmark سُنّي ثابت من 300–1000 سؤال مع إجابات ومصادر مراجعة بشرية.
3. اختر النموذج الفائز حسب صحة المصدر والرفض الآمن، وليس حسب fluency أو training loss فقط.
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_16][^2_3][^2_4][^2_5][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://www.promptquorum.com/power-local-llm/mobile-llm-models-phi4-gemma-smollm

[^2_2]: https://huggingface.co/RightNowAI/RightNow-Arabic-0.5B-Turbo

[^2_3]: https://arxiv.org/html/2601.21337v1

[^2_4]: https://arxiv.org/abs/2605.28827

[^2_5]: https://www.baseten.co/library/qwen-3-asr-1-7b/

[^2_6]: https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf

[^2_7]: https://github.com/QwenLM/Qwen3-ASR

[^2_8]: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B

[^2_9]: https://www.linkedin.com/posts/vadim-belsky_vadimbelskyqwen35-tts-emirati-hugging-activity-7438803487796289536-1Ds-

[^2_10]: https://www.linkedin.com/posts/mohammedfathy66_machinelearning-nlp-llm-activity-7486164934842617856-bJcz

[^2_11]: https://www.reddit.com/r/Rag/comments/1rrnse7/whats_the_best_and_most_popular_model_right_now/

[^2_12]: https://gist.github.com/unclecode/e5da5fb6a1d37022b089e243e0d9e00e

[^2_13]: https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai

[^2_14]: https://asas.ai/tools/arabic-llm-benchmark

[^2_15]: https://qwen-image-2512.com/blog/qwen3-asr-1.7b-complete-guide-en

[^2_16]: projects.islamic_llm.murad_sunni_schema


---

# @GitHub @Hugging Face عاوز داتا جاهزه ل Baligh-1.7B

أيوه، عندك داتا جاهزة ممتازة داخل Hugging Face، لكن **لا أنصح بتدريب Baligh-1.7B مباشرة على النصوص الخام**. الأفضل تعمل Dataset مركّبة: تعليمات عربية + أمثلة QA موثقة + بيانات RAG مع citations + أمثلة رفض آمن.

## أفضل الداتا الموجودة عندك

| الأولوية | Dataset | الاستخدام |
| :-- | :-- | :-- |
| 1 | [Athar-Datasets](https://huggingface.co/datasets/Kandil7/Athar-Datasets) | المصدر الرئيسي لبناء QA وتعليمات إسلامية |
| 2 | [Athar-Mini-Dataset-v2](https://huggingface.co/datasets/Kandil7/Athar-Mini-Dataset-v2) | بداية سريعة وتجارب Colab |
| 3 | [Athar-RAG-Hub](https://huggingface.co/datasets/Kandil7/Athar-RAG-Hub) | تدريب النموذج على الإجابة من السياق مع metadata |
| 4 | [Athar-Shamela4](https://huggingface.co/datasets/Kandil7/Athar-Shamela4) | Continued pretraining أو استخراج أمثلة عالية الجودة |
| 5 | [Tibyan Quran Complete](https://huggingface.co/datasets/Kandil7/tibyan-quran-complete) | آيات القرآن والمرجع الدقيق، وليس تدريب إجابات وحده |

نتيجة الفحص تُظهر أن `Athar-Datasets` يحتوي على حوالي 2.1 مليون صف متاح حاليًا، مع تقدير إجمالي يصل إلى 15.8 مليون passage، وحقول مثل `content`, `book_title`, `author`, `category`, `collection`, `page_number`, و`section_title`.[^3_1]

أما `Athar-RAG-Hub` فهو أنسب Dataset مباشرة لتدريب سلوك RAG؛ يحتوي على 5,852 chunk في مجموعة السيرة، مع metadata غنية مثل الكتاب، المؤلف، التصنيف، النص، المراجع القرآنية، collections الحديث، المذاهب، الإسناد، ودرجة الحديث عند توفرها.[^3_1]

## الـ Dataset المقترحة لـ 1.7B

أنشئ Dataset نهائية باسم:

```text
Kandil7/Baligh-SFT-1.7B
```

بصيغة ChatML أو conversational format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "أنت مساعد علمي عربي سني. لا تنسب قولًا إلى الله أو رسوله أو العلماء دون مصدر واضح. إذا لم يكفِ السياق، صرّح بعدم كفاية المعلومات."
    },
    {
      "role": "user",
      "content": "ما حكم المسألة المذكورة في النص؟"
    },
    {
      "role": "assistant",
      "content": "بحسب النص المسترجع، الحكم هو ...\n\nالمصدر: اسم الكتاب، المؤلف، الجزء، الصفحة."
    }
  ],
  "metadata": {
    "domain": "fiqh",
    "source_book": "...",
    "author": "...",
    "citation": "...",
    "answer_type": "grounded_qa",
    "sunni_review": true
  }
}
```

لا تجعل `metadata` جزءًا من النص الذي يراه النموذج إلا إذا كنت تريد تدريبه على إخراجها. الأفضل أن تستخدمها للفلترة، التقييم، وتقسيم البيانات.

## التركيبة العملية

لبداية قوية، استخدم تقريبًا:

```text
40% Athar-Datasets
25% Athar-RAG-Hub
15% MURAD-Sunni / المصطلحات والتعريفات
10% Quran + Hadith citation tasks
10% refusal / uncertainty / correction examples
```

لكن من `Athar-Datasets` لا تستخدم كل النصوص عشوائيًا. استخرج أمثلة من:

- العقيدة والتوحيد.
- التفسير وعلوم القرآن.
- الحديث مع التخريج والدرجة.
- الفقه المقارن مع ذكر المذهب والخلاف.
- السيرة.
- أصول الفقه.
- اللغة العربية لفهم النصوص التراثية.

`Athar-Shamela4` يحتوي على نحو 8,589 كتابًا وحوالي 7.6 مليون صفحة من النص العربي، ولذلك هو ممتاز كمصدر استخراج، لكنه ليس SFT dataset جاهزًا بصورته الخام.[^3_1]

## لا تستخدم هذه الطريقة

لا تحول كل passage إلى سؤال مصطنع بشكل آلي مثل:

```text
السؤال: ماذا يقول النص؟
الإجابة: [النص كاملًا]
```

هذا سيؤدي غالبًا إلى:

- حفظ النص بدل تعلم الإجابة.
- هلوسة أسئلة لا يجيب عنها المصدر.
- خلط كلام المؤلفين بالموقف الشرعي العام.
- ضعف في التمييز بين الحديث الصحيح والضعيف.
- إخفاء الخلاف الفقهي.
- إنتاج فتاوى قطعية من نصوص وصفية أو تاريخية.

الأصح أن تصنف كل عينة إلى نوع واضح:

```text
direct_quote
grounded_qa
definition
compare_positions
hadith_verification
citation_extraction
refusal_insufficient_context
madhhab_aware_fiqh
summarization
```


## مهم جدًا للمنهج السني

لا يكفي أن تكون الكتب إسلامية أو عربية. قبل إدخال العينة إلى SFT، أضف مرحلة مراجعة تتأكد من:

- هوية المؤلف ومنهجه.
- نوع المصدر: قرآن، حديث، تفسير، فقه، عقيدة، تاريخ.
- عدم تقديم نص منقول كأنه إجماع.
- عدم حذف قيد أو استثناء يغير الحكم.
- التمييز بين الحديث المرفوع والقول الفقهي.
- ذكر الخلاف عند وجوده.
- عدم الجزم بصحة الحديث دون مصدر أو حكم محدث.
- استخدام لغة مثل: «بحسب المصدر»، «ورد في النص»، «المسألة فيها خلاف».
- رفض الإجابة التفصيلية عندما لا يقدّم RAG مصدرًا كافيًا.

بالنسبة لأسئلة العقيدة والفتوى، اجعل المراجعة السنية البشرية إلزامية للعينة، ولا تعتمد على LLM-as-a-judge وحده.

## خطة تدريب مقترحة

### المرحلة الأولى: SFT

ابدأ بـ 100 ألف إلى 300 ألف عينة عالية الجودة، وليس ملايين الأمثلة منخفضة الجودة:

```yaml
base_model: Qwen/Qwen3-1.7B
dataset: Baligh-SFT-1.7B
method: QLoRA
max_seq_length: 4096
response_only_loss: true
learning_rate: 5e-5
lora_r: 32
lora_alpha: 64
epochs: 1-2
```


### المرحلة الثانية: preference alignment

بعد SFT، جهّز أمثلة:

```json
{
  "prompt": "...",
  "chosen": "إجابة موثقة ومتحفظة...",
  "rejected": "إجابة قطعية بلا مصدر أو فيها نسبة خاطئة..."
}
```

استخدمها في DPO أو ORPO، مع رفض الإجابات التي:

- تخترع مصدرًا.
- تنسب حديثًا بلا تخريج.
- تتجاهل النص المسترجع.
- تخلط بين المذاهب.
- تجيب بثقة عند نقص الدليل.


### المرحلة الثالثة: RAG behavior

درب النموذج على حالات فيها:

1. سياق كافٍ → يجيب ويذكر المصدر.
2. سياق متعارض → يوضح التعارض.
3. سياق غير ذي صلة → يرفض الاعتماد عليه.
4. لا يوجد سياق → يقول إنه لا يستطيع الجزم.
5. سؤال فتوى شخصية → يطلب التفاصيل أو يحيل إلى عالم موثوق.

## اختياري النهائي

لو تريد Dataset جاهزة الآن:

```text
Athar-Datasets + Athar-RAG-Hub
```

ولو تريد أسرع تجربة:

```text
Athar-Mini-Dataset-v2 + 10k–30k عينة مراجعة يدوية
```

ولو تريد بناء نسخة إنتاجية موثوقة:

```text
Athar-Datasets
+ Athar-RAG-Hub
+ MURAD-Sunni
+ Quran/Hadith citation set
+ curated refusal/preference set
```

ولا تستخدم `Athar-Shamela4` مباشرة كـ instruction tuning؛ استخدمه لاستخراج corpus للـ continued pretraining أو لبناء أمثلة QA منتقاة. كذلك توجد مشكلة تقنية ظاهرة في Dataset Viewer لبعض المستودعات: `Athar-Mini-Dataset-v2` فشل في توليد الـ split، و`Athar-Shamela4` لم يعرض config/split تلقائيًا، لذلك اختبر التحميل الفعلي عبر `datasets` قبل بناء pipeline الإنتاجي.[^3_1]

## Artifact

لا يوجد ملف مُنشأ.

## Next actions

1. ابنِ `Baligh-SFT-1.7B` من `Athar-Datasets` و`Athar-RAG-Hub` بعد تحويل الأمثلة إلى `messages`.
2. افصل train/validation حسب الكتاب والمؤلف، حتى لا يحدث تسريب بين نصوص المصدر.
3. جهّز 500 سؤالًا سنيًا مُراجعًا يدويًا لقياس صحة النقل، citations، الفقه، العقيدة، والرفض الآمن.

<div align="center">⁂</div>

[^3_1]: https://arxiv.org/html/2601.21337v1


---

# @GitHub @Hugging Face دور اكثر علي كل الداتا سيت الجاهزه فعليا

لقيت لك الداتا الجاهزة فعليًا، والنتيجة المهمة: **أقوى مصادر Baligh ليست Dataset واحدة مثالية، بل مجموعة Datasets تحتاج فلترة سنية ومراجعة citations قبل الـ fine-tuning**.

## 1. جاهزة مباشرة للـ SFT

### NightPrince/islamic-arabic-qa

[فتح Dataset](https://huggingface.co/datasets/NightPrince/islamic-arabic-qa)

- صيغة instruction-tuning.
- تغطي الفقه، الفتاوى، العقيدة، علوم القرآن، والتمويل الإسلامي.
- Parquet وملائمة لـ `datasets`.
- Apache-2.0.
- الحجم التقريبي: من 10 آلاف إلى أقل من 100 ألف عينة.

هذه أقرب Dataset جاهزة مباشرة لمهمة Baligh، لكن يجب مراجعة مصادر الإجابات ومنهجها السني قبل التدريب.[^4_1]

### Omar-youssef/islamic-qa-egyptian-arabic

[فتح Dataset](https://huggingface.co/datasets/Omar-youssef/islamic-qa-egyptian-arabic)

- 7,465 سؤالًا وجوابًا.
- باللهجة المصرية.
- تغطي الفقه والحديث والدراسات الإسلامية.
- Apache-2.0.
- Parquet ومناسبة للتحميل المباشر.

ممتازة لإضافة قدرة فهم المستخدم المصري، لكن لا تجعلها أساس Baligh؛ استخدمها تقريبًا بنسبة 5–10%، وافصل اللهجة عن إجابات الفصحى.[^4_1]

### AhmedBou/Arabic_instruction_dataset_for_llm_ft

[فتح Dataset](https://huggingface.co/datasets/AhmedBou/Arabic_instruction_dataset_for_llm_ft)

- Dataset عربية عامة للتعليمات.
- Parquet.
- مناسبة للحفاظ على قدرات المحادثة والتعليمات العامة.
- لا تبدو إسلامية متخصصة، لذلك استخدمها كـ general-domain mix وليس كبيانات دينية.


### akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft

[فتح Dataset](https://huggingface.co/datasets/akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft)

- حوالي 6 ملايين instruction sample بحسب الاسم والوصف.
- مناسبة للتوسيع العام وتحسين التنوع العربي.
- تحتاج sampling قويًا، ولا ينبغي إدخالها كاملة في Baligh حتى لا تطغى على السلوك السني.


## 2. بياناتك الجاهزة لـ RAG والتوليد

### Athar-Datasets

[فتح Dataset](https://huggingface.co/datasets/Kandil7/Athar-Datasets)

- مصدر إسلامي عربي كبير.
- يحتوي على قرآن، حديث، فقه، تفسير، عقيدة، وسيرة.
- الـ schema يتضمن `content`, `book_title`, `author`, `category`, `collection`, `page_number`, وبيانات القسم.
- يوجد حوالي 2.1 مليون صف ظاهر حاليًا، مع تقدير إجمالي أكبر يصل إلى 15.8 مليون passage.

استخدمه لاستخراج instruction examples وgrounded QA، وليس كـ SFT خام مباشرة.[^4_1]

### Athar-RAG-Hub

[فتح Dataset](https://huggingface.co/datasets/Kandil7/Athar-RAG-Hub)

هذه أفضل Dataset عندك لتعليم Baligh سلوك الإجابة من السياق:

- 5,852 chunk في مجموعة السيرة.
- تتضمن `text` و`book_title` و`author`.
- تحتوي metadata للمراجع القرآنية، كتب الحديث، المذاهب، الإسناد، ودرجة الحديث عند توفرها.
- مناسبة لبناء أمثلة: سؤال + context + answer + citation.

لكن الترخيص `CC-BY-NC-4.0`، لذلك راجع ملاءمته لأي استخدام تجاري.[^4_1]

### Athar-Mini-Dataset-v2

[فتح Dataset](https://huggingface.co/datasets/Kandil7/Athar-Mini-Dataset-v2)

- 100 ألف passage عربي إسلامي.
- موزعة على 10 collections.
- تشمل العقيدة واللغة وغيرها.
- مناسبة للتجربة الأولى على Colab.

يوجد حاليًا فشل في Dataset Viewer أثناء توليد الـ split، لذلك لا تفترض أنها قابلة للتحميل دون اختبار. افحص الملفات والـ README محليًا قبل إدخالها في pipeline.[^4_1]

### Athar-Shamela4

[فتح Dataset](https://huggingface.co/datasets/Kandil7/Athar-Shamela4)

- 8,589 كتابًا.
- حوالي 7.6 مليون صفحة.
- قرابة 19 GB من النص العربي.
- ممتازة للـ continued pretraining أو لاستخراج corpus وQA.

ليست SFT dataset جاهزة؛ هي corpus خام/منظم يحتاج chunking، تصنيف، attribution، واستخراج أمثلة. كما أن Dataset Viewer لم يستطع اكتشاف configs/splits تلقائيًا.[^4_1]

### Tibyan Quran Complete

[فتح Dataset](https://huggingface.co/datasets/Kandil7/tibyan-quran-complete)

- النص الكامل للقرآن.
- 114 سورة و6,236 آية.
- مناسبة لمهام رقم الآية، استرجاع الآية، المطابقة، والتوثيق.

لا تستخدمها لتعليم النموذج إنشاء تفسير من عنده؛ استخدمها لتدريب citation extraction وexact retrieval.

## 3. Datasets عربية مساعدة

| Dataset | فائدتها لـ Baligh | القرار |
| :-- | :-- | :-- |
| `Mohamed-Sami/instruction-fine-tuning-arabic-dataset` | تعليمات عربية عامة، حجم 100K–1M | استخدم عينة صغيرة |
| `Mohamed-Sami/arabic-instruction-fine-tuning-prep` | Dataset تحضيرية عربية | baseline فقط |
| `dispatchAI/Arabic-Mobile-Instructions` | تعليمات عربية للنماذج الصغيرة/on-device | مفيدة للـ deployment |
| `dispatchAI/arabic-poetry-instructions` | العربية الكلاسيكية والشعر | اختيارية لتحسين فهم النص التراثي |
| `pranav63/arabic-english-codeswitched-instructions` | Arabic-English code switching | 1–3% فقط إن كان مطلوبًا |

هذه النتائج ظهرت في بحث Hugging Face عن Arabic instruction datasets، لكنها ليست دينية بالضرورة، ولذلك لا تستخدمها في alignment السني كبيانات أساسية.

## 4. موارد GitHub القابلة للاستخدام

### Arabic-NLP-Data-Preparation-Pipeline

[فتح GitHub repository](https://github.com/moanayasser6/Arabic-NLP-Data-Preparation-Pipeline)

هذا ليس Dataset نهائيًا، لكنه pipeline لمعالجة بيانات الفتاوى العربية:

- توحيد ملفات Excel.
- معالجة القيم الناقصة.
- تطبيع النص العربي.
- حذف التكرار.
- توليد العناوين الناقصة.
- تجهيز البيانات للـ IR وNLP.

مفيد لبناء preprocessing layer قبل دمج مصادر الفتاوى، لكن افحص المصدر الأصلي وحقوق الاستخدام قبل الاعتماد عليه.

### AraIslaMorals

[فتح GitHub repository](https://github.com/Arwaalmrzoqi/AraIslaMorals)

Dataset/مشروع لتقييم توافق النماذج مع القيم الأخلاقية العربية الإسلامية. لا أضعه كبيانات SFT أساسية، لكنه مفيد لبناء evaluation set للسلوك والقيم والرفض.

### JSON-Quran

[فتح GitHub repository](https://github.com/ANAS-999/JSON-Quran)

ملف قرآن JSON منظم مع metadata للسور بالعربية والإنجليزية. مفيد كـ auxiliary exact-match/evaluation resource، مع ضرورة التحقق من مصدر النص والترخيص قبل الدمج.

## 5. الـ stack الذي أنصح به

لنسخة `Baligh-1.7B`، ابدأ بهذا الترتيب:

```text
60%  NightPrince/islamic-arabic-qa
10%  Omar-youssef/islamic-qa-egyptian-arabic
15%  Athar-RAG-Hub
10%  Athar-Datasets بعد تحويله إلى grounded QA
5%   Arabic general instruction data
```

لكن هذه النسب **قبل الفلترة**. بعد مراجعة العينة، قد يصبح الحجم النهائي أصغر بكثير، وهذا أفضل من تدريب النموذج على إجابات دينية غير موثقة.

لـ SFT عملي:

```python
from datasets import load_dataset

ds = load_dataset(
    "NightPrince/islamic-arabic-qa",
    split="train"
)

print(ds)
print(ds.column_names)
print(ds[^4_0])
```

وبالنسبة لـ Athar، لا تدخل `content` كما هو. حوّله إلى أمثلة من النوع:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "أجب وفق النص المرفق، ولا تنسب قولًا بلا مصدر."
    },
    {
      "role": "user",
      "content": "السؤال: ...\n\nالسياق:\n..."
    },
    {
      "role": "assistant",
      "content": "الإجابة: ...\n\nالمصدر: ..."
    }
  ]
}
```


## الترتيب النهائي

### جاهزة للبدء فورًا

1. `NightPrince/islamic-arabic-qa`
2. `Omar-youssef/islamic-qa-egyptian-arabic`
3. `AhmedBou/Arabic_instruction_dataset_for_llm_ft`

### الأفضل لبناء النسخة السنية الموثقة

1. `Athar-RAG-Hub`
2. `Athar-Datasets`
3. `Tibyan Quran Complete`
4. `MURAD-Sunni` من بيانات مشروعك
5. مجموعة رفض وتصحيح ومقارنة مواقف فقهية من إعدادك

### لا تستخدم مباشرة

- `Athar-Shamela4` الخام.
- ملايين instruction samples دون فحص.
- أي Dataset فتوى لا تحتوي attribution واضحًا.
- بيانات مولدة آليًا دون مراجعة سُنّية.
- كل مصدر إسلامي على أنه يمثل منهج أهل السنة تلقائيًا.


## الخلاصة

لو تريد تشغيل تجربة اليوم، استخدم:

```text
NightPrince/islamic-arabic-qa
+ Omar-youssef/islamic-qa-egyptian-arabic
+ Athar-RAG-Hub
```

ولو تريد إنتاج Baligh موثوقًا، اجعل `Athar` مصدر المعرفة، واجعل Dataset الـ SFT من **أمثلة QA موثقة ومراجعة**، لا من passages خام. هذه النقطة مهمة لأن النموذج يجب أن يتعلم متى يجيب، ومتى يذكر الخلاف، ومتى يقول «لا أعلم» أو «السياق غير كافٍ»، لا أن يحفظ نصوصًا كثيرة فقط.

## Artifact

لا يوجد ملف مُنشأ.

## Next actions

1. نزّل وافحص schema وsample rows من `NightPrince/islamic-arabic-qa` و`Omar-youssef/islamic-qa-egyptian-arabic`.
2. حوّل `Athar-RAG-Hub` إلى ChatML مع citation fields ووسوم `answer_type`.
3. اعمل audit يدوي على 500 عينة قبل تدريب Baligh-1.7B، مع استبعاد أي إجابة بلا مصدر أو بمنهج غير واضح.
<span style="display:none">[^4_2]</span>

<div align="center">⁂</div>

[^4_1]: https://arxiv.org/html/2601.21337v1

[^4_2]: projects.islamic_llm.murad_sunni_schema


---

# @GitHub @Hugging Face عاوز اسكربت كامل يجمع الداتا و يجهزها للمرحله النهائيه ل fine tuning  علي Qwen3-1.7B لانشاء Baligh-1.7B

جهزت لك سكربت كامل باسم **`prepare_baligh_qwen3_17b.py`**. السكربت يقوم بالآتي:

- يجمع البيانات من:
    - `NightPrince/islamic-arabic-qa`
    - `Omar-youssef/islamic-qa-egyptian-arabic`
    - `AhmedBou/Arabic_instruction_dataset_for_llm_ft`
    - `Kandil7/Athar-Datasets`
    - `Kandil7/Athar-RAG-Hub`
- يكتشف أسماء الأعمدة الشائعة تلقائيًا.
- يحول الأمثلة إلى صيغة `messages` المتوافقة مع Qwen3.
- يضيف system prompt خاصًا بـ Baligh ومنهج أهل السنة.
- يطبع metadata للمصدر والتصنيف والـ citation.
- يطبّع النص العربي ويزيل التكرارات.
- ينشئ `train` و`validation`.
- يفصل passages الخام التي تحتاج مراجعة في `needs_review.jsonl`.
- يحفظ البيانات بصيغة JSONL وHugging Face `save_to_disk`.
- يدعم الرفع إلى Hugging Face Dataset خاص.


## التشغيل

ثبت المتطلبات:

```bash
pip install -U datasets huggingface_hub pyarrow
```

سجّل الدخول:

```bash
huggingface-cli login
```

أو:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

شغّل تجربة صغيرة أولًا:

```bash
python prepare_baligh_qwen3_17b.py \
  --max-per-source 1000 \
  --output-dir data/baligh_qwen3_1.7b_smoke
```

ثم شغّل النسخة الكاملة:

```bash
python prepare_baligh_qwen3_17b.py \
  --output-dir data/baligh_qwen3_1.7b \
  --validation-ratio 0.02 \
  --seed 42
```

ولرفعها إلى Hugging Face:

```bash
python prepare_baligh_qwen3_17b.py \
  --output-dir data/baligh_qwen3_1.7b \
  --push-to-hub Kandil7/Baligh-SFT-Qwen3-1.7B
```


## تنبيه مهم

السكربت لا يخترع إجابات دينية من passages الخام. أي passage لا يحتوي سؤالًا وإجابة جاهزين يُحفظ في:

```text
data/baligh_qwen3_1.7b/needs_review.jsonl
```

وهذا مقصود؛ لا ينبغي تدريب Baligh على إجابات مولدة آليًا أو تلخيصات غير مراجعة في العقيدة والفقه والحديث.

الداتا الأصلية `Athar-Datasets` و`Athar-RAG-Hub` تحتوي metadata مهمة للمصادر، مثل اسم الكتاب والمؤلف والتصنيف والصفحة، و`Athar-RAG-Hub` يضم أيضًا حقولًا للمراجع القرآنية وكتب الحديث والمذاهب والإسناد عند توفرها.  أما مستودع Baligh نفسه فيحتوي بالفعل على تقسيمات إعداد منفصلة لـ SFT، لذلك يمكن لاحقًا ربط ناتج السكربت بملفات التدريب الموجودة في المشروع.[^5_1]

### ملاحظة تقنية

في النسخة الحالية، الـ passages الخام تُفصل للمراجعة ولا تدخل `train` تلقائيًا. هذا أكثر أمانًا، لكنه يعني أن `train` الأولي يعتمد على Datasets التي تحتوي أصلًا على سؤال/إجابة. بعد مراجعة `needs_review.jsonl`، حوّل العينات المقبولة إلى إجابات حقيقية ثم أعد تشغيل مرحلة الدمج.

## Artifact

الملف البرمجي الجاهز: **`prepare_baligh_qwen3_17b.py`**

## Next actions

1. شغّل نسخة smoke test بـ `--max-per-source 1000` وافحص أسماء الأعمدة والعينات.
2. راجع `needs_review.jsonl` سنيًا، خصوصًا العقيدة والفتاوى ونسبة الأحاديث.
3. بعد تثبيت الـ schema، اربط الناتج بمرحلة QLoRA على `Qwen/Qwen3-1.7B` مع فصل validation حسب المصدر والكتاب.
<span style="display:none">[^5_2]</span>

<div align="center">⁂</div>

[^5_1]: https://arxiv.org/html/2601.21337v1

[^5_2]: projects.islamic_llm.murad_sunni_schema


---

# @GitHub @Hugging Face [https://huggingface.co/datasets/Raniahossam33/Islamweb_part2](https://huggingface.co/datasets/Raniahossam33/Islamweb_part2)

الداتا دي **مناسبة جدًا كمصدر SFT لـ Baligh-1.7B**، وأفضل من الـ passages الخام؛ لأنها تحتوي على سؤال، سياق السؤال، وإجابة كاملة. لكنها ليست جاهزة للدمج بدون تنظيف ومراجعة، خصوصًا لأنها فتاوى من موقع واحد وليست بالضرورة ممثلة لكل مسائل أهل السنة أو لكل المذاهب.

## مواصفات Dataset

`Raniahossam33/Islamweb_part2` تحتوي على:

- حوالي **73,600 عينة**.
- صيغة CSV/Parquet.
- الأعمدة:
    - `Question Number`
    - `URL`
    - `Question`
    - `Question_Context`
    - `Answer`
- روابط أصلية لصفحات الفتاوى، وهي مفيدة جدًا لحفظ attribution.
- Dataset Viewer يعرض أمثلة في الفقه، الحديث، العقيدة، المواريث، الطهارة، الأسرة، وغيرها.[^6_1]

مثال السجل مناسب جدًا للتحويل إلى:

```text
system → user question + context → assistant answer + source URL
```


## هل تدخل مباشرة؟

نعم، لكن بهذه السياسة:


| الاستخدام | القرار |
| :-- | :-- |
| SFT على أسلوب السؤال والجواب | مناسب |
| تدريب citations وذكر المصدر | مناسب جدًا |
| اعتبار الإجابات حقائق نهائية | غير آمن |
| continued pretraining | غير مناسب |
| تدريب النموذج على النص الخام فقط | لا |
| تدريب الفتوى دون مراجعة | لا |

السبب أن بعض الإجابات تعرض قولًا فقهيًا محددًا أو ترجيحًا خاصًا، وبعضها يذكر خلافًا بين الجمهور ومذهب معين. يجب أن يتعلم Baligh أن يقول: «بحسب الفتوى المذكورة» أو «هذا قول اختارته الجهة الناقلة»، بدل عرض كل جواب على أنه إجماع.

## التحويل المقترح

صيغة `messages`:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "أنت بليغ، مساعد عربي علمي ملتزم بمنهج أهل السنة والجماعة. أجب اعتمادًا على السياق والمصدر، ولا تنسب قولًا بلا توثيق. عند وجود خلاف معتبر اذكره، وإذا لم يكف السياق فقل لا أعلم أو لا يكفي السياق للجزم."
    },
    {
      "role": "user",
      "content": "السؤال:\n...\n\nتفاصيل السائل:\n..."
    },
    {
      "role": "assistant",
      "content": "...\n\nالمصدر: إسلام ويب\nالرابط: ..."
    }
  ],
  "metadata": {
    "source": "Islamweb",
    "source_url": "...",
    "question_number": 165945,
    "domain": "fiqh",
    "needs_review": true
  }
}
```


## كود تحميل وتحويل خاص بها

أضف هذا المصدر إلى السكربت السابق بدل الاعتماد على الاكتشاف العام للأعمدة:

```python
from datasets import load_dataset

repo_id = "Raniahossam33/Islamweb_part2"
ds = load_dataset(repo_id, split="train")

SYSTEM = (
    "أنت بليغ، مساعد عربي علمي ملتزم بمنهج أهل السنة والجماعة. "
    "أجب اعتمادًا على السياق والمصدر، ولا تنسب قولًا بلا توثيق. "
    "عند وجود خلاف معتبر اذكره، وإذا لم يكف السياق فقل: لا أعلم، "
    "أو لا يكفي السياق للجزم. في مسائل الفتوى الشخصية نبّه إلى مراجعة عالم موثوق."
)


def make_islamweb_example(row):
    question = str(row["Question"]).strip()
    context = str(row["Question_Context"]).strip()
    answer = str(row["Answer"]).strip()
    url = str(row["URL"]).strip()
    number = int(row["Question Number"])

    user = f"السؤال:\n{question}"
    if context and context.lower() != "nan":
        user += f"\n\nتفاصيل السائل:\n{context}"

    assistant = f"{answer}\n\nالمصدر: إسلام ويب، فتوى رقم {number}\nالرابط: {url}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source_dataset": repo_id,
        "source_url": url,
        "question_number": number,
        "example_type": "fatwa_qa",
        "needs_sunni_review": True,
    }


converted = ds.map(
    make_islamweb_example,
    remove_columns=ds.column_names,
    desc="Converting Islamweb to ChatML",
)
```


## التنظيف الضروري

قبل إضافتها إلى `train`:

- احذف العينات التي يكون فيها `Question` أو `Answer` فارغًا.
- احذف HTML والرموز الزائدة.
- وحّد المسافات والتشكيل، لكن احتفظ بنسخة أصلية للرجوع.
- أزل التكرارات حسب السؤال normalized.
- افحص الإجابات القصيرة جدًا.
- افصل الفتاوى التي تحتوي على أكثر من مسألة.
- احتفظ برقم الفتوى والرابط.
- لا تحذف عبارات مثل «اختلف العلماء» أو «والله أعلم».
- لا تحذف إحالات الفتوى الداخلية مثل «راجع الفتوى رقم...».
- افصل الفتاوى الطبية والقانونية والمالية في domains مستقلة.
- لا تجعل `URL` جزءًا من السؤال؛ ضعه في metadata والإجابة فقط إذا كنت تريد تعليم citation output.


## تقسيم train/validation

لا تعمل split عشوائي فقط؛ لأن نفس الموضوع قد يتكرر بصيغ مختلفة. استخدم:

```text
90–95% train
5–10% validation
```

والأفضل تقسيم حسب `Question Number` أو URL، مع منع وجود أسئلة متشابهة جدًا في الطرفين.

مثال:

```python
from datasets import DatasetDict

split = converted.train_test_split(test_size=0.05, seed=42)
final_ds = DatasetDict(
    {
        "train": split["train"],
        "validation": split["test"],
    }
)
```

لكن للتقييم الحقيقي، أنشئ validation منفصلًا يدويًا من موضوعات لم تظهر في التدريب، مثل:

- الطهارة والصلاة.
- الصيام.
- المواريث.
- العقيدة.
- الحديث والتخريج.
- الطلاق والأسرة.
- المعاملات.
- الأسئلة التي لا يكفي فيها السياق.


## مكانها في Dataset النهائية

أقترح أن تكون:

```text
20–30% من SFT dataset
```

وليس 100% من التدريب، لأن Baligh يحتاج أيضًا إلى:

- العربية العامة.
- RAG grounded answers من Athar.
- MURAD-Sunni definitions.
- أمثلة الرفض وعدم الجزم.
- أمثلة مقارنة الأقوال والمذاهب.
- citations للقرآن والحديث.
- بيانات مصرية محدودة لفهم اللهجة.

تركيبة أفضل:

```text
25% Islamweb_part2
25% Athar-RAG-Hub grounded QA
20% Athar-Datasets curated QA
15% MURAD-Sunni / definitions
10% Arabic general instructions
5% refusal and uncertainty examples
```


## نقطة منهجية مهمة

إسلام ويب مصدر دعوي وفتاوى، لكنه ليس بديلًا عن corpus مصنف للمصادر الأصلية. استخدمه لتعليم:

- نمط الإجابة.
- تنظيم الفتوى.
- السؤال والجواب.
- الإحالة للمصدر.
- التعامل مع الحالات الشخصية.

واستخدم `Athar` وكتب العلماء الأصلية لتعليم:

- النصوص الأصلية.
- التعريفات.
- التخريج.
- التصنيف.
- المذاهب.
- مراتب الأدلة.

لا تجعل النموذج يحفظ إجابات إسلام ويب كأنها مصدر مستقل عن الكتب والأدلة التي تحيل إليها.

## الحكم النهائي

نعم، أضف Dataset إلى pipeline، وأعطها وسم:

```text
source = islamweb
example_type = fatwa_qa
needs_sunni_review = true
```

وهي مناسبة جدًا للـ SFT بعد التنظيف، لكن لا تدخل النسخة النهائية قبل مراجعة بشرية لعينة ممثلة. Dataset تحتوي 73.6 ألف سجل بخمسة أعمدة واضحة، وهذا يجعل تحويلها إلى ChatML مباشرًا نسبيًا.

## Artifact

لا يوجد ملف جديد؛ التحويل المقترح مضمّن في الرد، ويمكن دمجه داخل `prepare_baligh_qwen3_17b.py`.

## Next actions

1. أضف `Raniahossam33/Islamweb_part2` إلى قائمة مصادر السكربت كـ `fatwa_qa`.
2. حوّل الـ 73.6 ألف سجل إلى ChatML، ثم افصلها عن `needs_review`.
3. راجع يدويًا 500–1000 عينة مع تصنيف المجال، المصدر، المذهب، وجود الخلاف، وصحة citation.
<span style="display:none">[^6_2]</span>

<div align="center">⁂</div>

[^6_1]: https://huggingface.co/datasets/Raniahossam33/Islamweb_part2

[^6_2]: projects.islamic_llm.murad_sunni_schema


---

# دور اكثر علي كل الداتا سيت الجاهزه فعليا

بحثت بشكل أوسع في Hugging Face وGitHub. الخلاصة: **لا توجد Dataset عامة واحدة جاهزة ومضمونة تصلح وحدها لتدريب Baligh-1.7B**، لكن توجد مجموعات يمكن استخدامها مباشرة بعد تحويل schema وتنظيف ومراجعة سنية.

## أفضل Datasets دينية جاهزة

| Dataset | الحجم/الصيغة | الملاءمة لـ Baligh |
| :-- | --: | :-- |
| [Raniahossam33/Islamweb_part2](https://huggingface.co/datasets/Raniahossam33/Islamweb_part2) | 73.6K، سؤال/سياق/إجابة/URL | ممتازة لـ SFT الفقهي، تحتاج مراجعة |
| [NightPrince/islamic-arabic-qa](https://huggingface.co/datasets/NightPrince/islamic-arabic-qa) | 10K–100K، Parquet | أفضل مصدر ديني مباشر للـ instruction tuning |
| [Omar-youssef/islamic-qa-egyptian-arabic](https://huggingface.co/datasets/Omar-youssef/islamic-qa-egyptian-arabic) | 7,465، Parquet | مفيدة للهجة المصرية، بنسبة صغيرة |
| [Kandil7/Athar-RAG-Hub](https://huggingface.co/datasets/Kandil7/Athar-RAG-Hub) | 5,852 chunk | ممتازة لتعليم grounded RAG |
| [Kandil7/Athar-Datasets](https://huggingface.co/datasets/Kandil7/Athar-Datasets) | ملايين passages | مصدر استخراج، وليس SFT خام |
| [Kandil7/Athar-Mini-Dataset-v2](https://huggingface.co/datasets/Kandil7/Athar-Mini-Dataset-v2) | 100K passage | تجربة أولية، لكن Dataset Viewer فيه مشكلة |
| [Kandil7/Athar-Shamela4](https://huggingface.co/datasets/Kandil7/Athar-Shamela4) | 8,589 كتابًا | continued pretraining/extraction، وليس SFT مباشر |
| [Kandil7/tibyan-quran-complete](https://huggingface.co/datasets/Kandil7/tibyan-quran-complete) | 6,236 آية | citations، exact retrieval، وتقييم |

`Islamweb_part2` هي الأكثر جاهزية تقنيًا من حيث schema؛ تحتوي على `Question`, `Question_Context`, `Answer`, و`URL`، وعددها حوالي 73.6 ألف سجل.  أما `Athar-RAG-Hub` فتحتوي metadata غنية تشمل الكتب والمؤلفين والمراجع القرآنية وكتب الحديث والمذاهب والإسناد عند توفرها، لذلك هي أنسب لسلوك RAG وليس للتدريب الحواري وحده.[^7_1]

## أفضل Datasets عربية عامة

هذه لا تدرّب المنهج السني، لكنها تساعد في الحفاظ على جودة العربية واتباع التعليمات:

- [riotu-lab/ArabicQA_2.1M](https://huggingface.co/datasets/riotu-lab/ArabicQA_2.1M): حوالي 2.1M سؤال وجواب عربي، Apache-2.0.
- [bobez999/arabic-qa-dataset-sigir2024](https://huggingface.co/datasets/bobez999/arabic-qa-dataset-sigir2024): 10K instruction-tuning pairs، MIT.
- [Mohamed-Sami/instruction-fine-tuning-arabic-dataset](https://huggingface.co/datasets/Mohamed-Sami/instruction-fine-tuning-arabic-dataset): 100K–1M، Parquet.
- [AhmedBou/Arabic_instruction_dataset_for_llm_ft](https://huggingface.co/datasets/AhmedBou/Arabic_instruction_dataset_for_llm_ft): بيانات تعليمات عربية عامة.
- [akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft](https://huggingface.co/datasets/akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft): حجم كبير، لكن يجب sampling قوي.
- [MohammedNasser/Arabic_Reasoning_Instruct_QA](https://huggingface.co/datasets/MohammedNasser/Arabic_Reasoning_Instruct_QA): reasoning عربي.
- [HeshamHaroon/QA_Arabic](https://huggingface.co/datasets/HeshamHaroon/QA_Arabic): QA عربي صغير.
- [dispatchAI/Arabic-Mobile-Instructions](https://huggingface.co/datasets/dispatchAI/Arabic-Mobile-Instructions): تعليمات عربية للنماذج الصغيرة.
- [Pangeanic/Iraqi-Arabic-multidomain-QA-text](https://huggingface.co/datasets/Pangeanic/Iraqi-Arabic-multidomain-QA-text): حوارات باللهجة العراقية.
- [dispatchAI/arabic-poetry-instructions](https://huggingface.co/datasets/dispatchAI/arabic-poetry-instructions): العربية الكلاسيكية والشعر.

ظهر في البحث أن `ArabicQA_2.1M` و`bobez999/arabic-qa-dataset-sigir2024` هما من أكبر الخيارات الجاهزة بصيغة QA/Instruction، لكن لا ينبغي خلطهما بكثافة مع بيانات Baligh الدينية حتى لا يضعف الالتزام بالمصادر.

## Datasets للتقييم والحماية

### Arabic-Hallucination-QA

[فتح Dataset](https://huggingface.co/datasets/Lameesalturki/Arabic-Hallucination-QA)

استخدمها للتقييم، وليس بالضرورة للتدريب، لمعرفة قدرة Baligh على كشف الإجابة غير الموثوقة.

### AraIslaMorals

[فتح GitHub repository](https://github.com/Arwaalmrzoqi/AraIslaMorals)

مفيد لتقييم القيم والأخلاق العربية الإسلامية، وليس مصدر SFT ديني رئيسيًا.

### AM-RAGBench

[فتح GitHub repository](https://github.com/Akramtaha98/AM-RAGBench-benchmark)

Benchmark للتحقق من faithfulness في RAG، ويحتوي على QA مع تقييم بشري. استخدم الفكرة والمنهجية لبناء benchmark خاص بـ Athar، مع عدم افتراض أن نطاقه الديني يكفي لتقييم Baligh السني.

### Masader

[فتح GitHub repository](https://github.com/ARBML/masader)

أهم مورد للبحث المنهجي؛ هو catalog يضم أكثر من 500 Dataset عربية مع خصائص متعددة. استخدمه لاكتشاف datasets إضافية حسب المهمة واللغة والترخيص بدل البحث النصي فقط.

## Datasets مساعدة متخصصة

لا تدخل هذه في SFT الديني الأساسي، لكنها مفيدة لوظائف محددة:

- [Sima](https://github.com/NoorBayan/Sima): تصنيف التشبيهات البلاغية في القرآن؛ مفيدة للتقييم أو auxiliary task.
- [Ifta-Sunnah-Hadith-and-Narrators-Dataset](https://github.com/JehadOumer/Ifta-Sunnah-Hadith-and-Narrators-Dataset): بيانات حديث ورواة وإسناد؛ تحتاج فحصًا دقيقًا للمصدر والترخيص.
- [quran-dataset](https://github.com/mjmirza/quran-dataset): قرآن عربي منظم مع metadata؛ CC BY 4.0 بحسب مستودع GitHub.
- [quran-etl](https://github.com/ezAldinWaez/quran-etl): pipeline ينتج JSON موثقًا وجاهزًا للـ LLM من مصادر Tanzil.
- [Quran-Tafseer-RAG](https://github.com/OmarEssameldinMousa/Quran-Tafseer-RAG): مشروع RAG للتفسير العربي مع dataset قابلة للاستخدام بعد فحص المصدر.
- [Arabic-NLP-Data-Preparation-Pipeline](https://github.com/moanayasser6/Arabic-NLP-Data-Preparation-Pipeline): pipeline لتنظيف وتجهيز بيانات الفتاوى العربية.
- [ArMATH](https://github.com/reem-codes/ArMATH): مسائل رياضية عربية؛ auxiliary فقط، وليست ضمن المجال الإسلامي.


## Dataset النهائية المقترحة

ابدأ بـ **نسخة عالية الجودة من 150K–300K عينة**، لا بكل البيانات المتاحة:

```text
25% Islamweb_part2
20% NightPrince/islamic-arabic-qa
15% Athar-RAG-Hub
15% Athar-Datasets بعد استخراج QA فقط
10% MURAD-Sunni
5% Omar-youssef/islamic-qa-egyptian-arabic
5% ArabicQA عامة
5% رفض، عدم يقين، وتصحيح citations
```

ولا تدخل `Athar-Shamela4` الخام في SFT. استخدمه فقط في:

```text
continued pretraining
استخراج أسئلة وأجوبة
تدريب citation extraction
بناء corpus للـ RAG
```


## ما أعتبره جاهزًا فعليًا؟

### جاهز تقنيًا

- `Islamweb_part2`.
- `NightPrince/islamic-arabic-qa`.
- `Omar-youssef/islamic-qa-egyptian-arabic`.
- `ArabicQA_2.1M`.
- `bobez999/arabic-qa-dataset-sigir2024`.
- `Mohamed-Sami/instruction-fine-tuning-arabic-dataset`.


### جاهز بعد تحويل وصناعة أمثلة

- `Athar-RAG-Hub`.
- `Athar-Datasets`.
- `Tibyan Quran Complete`.
- بيانات القرآن والحديث والإسناد من GitHub.


### غير جاهز للتدريب المباشر

- `Athar-Shamela4`.
- أي corpus كتب خام.
- أي dataset بدون attribution.
- أي dataset مولدة آليًا بدون مراجعة.
- أي مصدر لا تستطيع تحديد ترخيصه.


## القرار العملي

أفضل مجموعة تبدأ بها الآن:

```text
Raniahossam33/Islamweb_part2
NightPrince/islamic-arabic-qa
Kandil7/Athar-RAG-Hub
Omar-youssef/islamic-qa-egyptian-arabic
bobez999/arabic-qa-dataset-sigir2024
```

لكن قبل التدريب النهائي، يجب أن تنتج manifest يحتوي على:

```text
source_dataset
source_url
book_title
author
category
madhhab
hadith_grade
citation
license
needs_sunni_review
quality_score
split
```

ومن الناحية العلمية، لا تعتبر أي Dataset «سنية» لمجرد أنها إسلامية؛ لا بد من مراجعة مصادر العقيدة والحديث والفتوى، وحفظ نسبة القول إلى قائله، وبيان الخلاف المعتبر. هذا هو الفارق بين Baligh موثّق وchatbot ديني يحفظ إجابات متفرقة.

## Artifact

لا يوجد ملف مُنشأ.

## Next actions

1. استخدم `Masader` كفهرس لاكتشاف datasets عربية إضافية، ثم أضف فقط المصادر ذات الترخيص والـ schema الواضحين.
2. حمّل وفحص schema لأول خمس Datasets في قسم «القرار العملي»، ثم وحّدها في format واحد.
3. أنشئ manifest ومراجعة بشرية سُنّية قبل إدخال أي عينة إلى تدريب Qwen3-1.7B.
<span style="display:none">[^7_2]</span>

<div align="center">⁂</div>

[^7_1]: https://arxiv.org/html/2601.21337v1

[^7_2]: projects.islamic_llm.murad_sunni_schema


---

# عاوز اسكربت كامل يجمع الداتا و يجهزها للمرحله النهائيه ل fine tuning  علي Qwen3-1.7B لانشاء Baligh-1.7B

جهزت لك سكربتًا كاملًا باسم **`prepare_baligh_final_qwen3_17b.py`** لتجهيز بيانات SFT النهائية لـ `Qwen/Qwen3-1.7B`.

## ما الذي يفعله؟

يجمع ويحوّل هذه المصادر:

- `Raniahossam33/Islamweb_part2`
- `NightPrince/islamic-arabic-qa`
- `Omar-youssef/islamic-qa-egyptian-arabic`
- `bobez999/arabic-qa-dataset-sigir2024`
- `Kandil7/Athar-RAG-Hub`

وتبيّن فحص Hugging Face أن:

- `NightPrince/islamic-arabic-qa` لديها 17,944 train و2,101 validation و1,042 test، وبصيغة `messages`, `system`, `question`, `answer`.
- `Islamweb_part2` لديها 73.6K سجلًا بأعمدة سؤال، سياق، إجابة، ورابط.
- `Omar-youssef/islamic-qa-egyptian-arabic` لديها 7,465 سؤالًا وجوابًا باللهجة المصرية.
- `ArabicQA_2.1M` لديها 2.1M سجلًا، لكن لم أضعها في السكربت الأساسي حتى لا تطغى بيانات عامة غير دينية على alignment الخاص بـ Baligh.
- `Athar-RAG-Hub` يحتوي على 5,852 chunk مع metadata دينية ومصدرية واسعة، لكنه يستخدم ترخيص `CC-BY-NC-4.0` ويجب مراجعة ملاءمته للاستخدام التجاري.


## المخرجات

بعد التشغيل سينشئ:

```text
data/baligh_final_qwen3_1.7b/
├── train.jsonl
├── validation.jsonl
├── needs_review.jsonl
└── manifest.json
```

- `train.jsonl`: عينات جاهزة تقنيًا بصيغة `messages`.
- `validation.jsonl`: validation مقسّمة على مستوى المصدر.
- `needs_review.jsonl`: فتاوى وRAG passages تحتاج مراجعة بشرية.
- `manifest.json`: المصادر، أعداد العينات، التراخيص، والإحصائيات.


## التشغيل

```bash
pip install -U datasets huggingface_hub pyarrow
```

ثم:

```bash
huggingface-cli login
```

اختبار سريع:

```bash
python prepare_baligh_final_qwen3_17b.py \
  --limit-per-source 1000 \
  --output-dir data/baligh_smoke
```

تشغيل كامل:

```bash
python prepare_baligh_final_qwen3_17b.py \
  --output-dir data/baligh_final_qwen3_1.7b \
  --validation-ratio 0.05 \
  --seed 42
```


## تحذير مهم

السكربت يحوّل البيانات ويجهزها تقنيًا، لكنه لا يدّعي أن كل الإجابات صحيحة عقديًا أو فقهيًا. لذلك:

- `Islamweb_part2` تدخل قائمة المراجعة لأنها فتاوى.
- `Athar-RAG-Hub` يدخل قائمة المراجعة لأن الـ passages ليست إجابات SFT جاهزة.
- البيانات العامة لا تُعامل كمصادر سنية.
- لا تستخدم `needs_review.jsonl` في التدريب قبل صياغة الإجابة ومراجعتها.
- احتفظ بالمصدر الأصلي والـ URL ورقم السؤال.
- لا تخلط train مع validation من نفس السؤال أو URL.

ولقد راجعت هيكل مشروع Baligh؛ توجد بالفعل طبقات منفصلة للبيانات مثل `cleaner.py`, `datasets.py`, `formatter.py`, `loader.py`, `mixer.py`, و`validators.py`، لذلك يمكن نقل منطق السكربت لاحقًا إلى هذه الوحدات بدل إبقائه standalone.

## ملاحظة على الكود

الإصدار الحالي لا يضع تلقائيًا إجابات مولدة لـ `Athar-RAG-Hub` داخل التدريب، وهذا قرار مقصود. يجب أولًا تحويل كل passage إلى:

```text
question + context + reviewed answer + citation
```

حتى لا يتعلم النموذج اختلاق إجابة أو تلخيصًا غير موثق.

## Artifact

الملف البرمجي الجاهز: **`prepare_baligh_final_qwen3_17b.py`**

## Next actions

1. شغّل smoke test بـ `--limit-per-source 1000` وافحص `manifest.json` و`train.jsonl`.
2. راجع `needs_review.jsonl` سنيًا، خصوصًا العقيدة، الحديث، الفقه، والفتاوى الشخصية.
3. بعد اعتماد البيانات، اربط `train.jsonl` و`validation.jsonl` بمرحلة QLoRA على `Qwen/Qwen3-1.7B`.
<span style="display:none">[^8_1]</span>

<div align="center">⁂</div>

[^8_1]: projects.islamic_llm.murad_sunni_schema


---

# ازاي اعمل  fine tuning  علي Qwen3-1.7B لانشاء Baligh-1.7B

لإنشاء **Baligh-1.7B**، استخدم `Qwen/Qwen3-1.7B` مع **QLoRA + SFT** على بيانات `messages`، مع حساب الـ loss على ردود المساعد فقط. النموذج الرسمي هو `AutoModelForCausalLM` بحوالي 2.03B parameters، مبني على `Qwen3-1.7B-Base`، وترخيصه Apache-2.0.

## 1. تجهيز البيئة

على Colab أو Linux:

```bash
pip install -U \
  torch \
  transformers \
  datasets \
  accelerate \
  peft \
  bitsandbytes \
  trl \
  huggingface_hub \
  sentencepiece
```

سجّل الدخول:

```bash
huggingface-cli login
```

وتأكد أن لديك GPU. للتجربة، GPU بذاكرة 16GB قد يعمل مع batch صغير وQLoRA، بينما 24GB أكثر راحة. بعض البيئات السحابية توصي بـ A10 24GB أو أعلى لتدريب Qwen3، لكن المتطلبات الفعلية تعتمد على sequence length وbatch size وgradient checkpointing.[^9_1]

## 2. شكل البيانات

استخدم `train.jsonl` و`validation.jsonl` بهذا الشكل:

```json
{"messages":[{"role":"system","content":"أنت بليغ..."},{"role":"user","content":"ما حكم ...؟"},{"role":"assistant","content":"بحسب المصدر ..."}]}
```

لا تضع `metadata` داخل نص المحادثة إلا إذا كنت تريد أن يتعلم النموذج إخراجها. احتفظ بها للتحقق والتقييم.

تحميل Dataset الناتجة من سكربت التجهيز:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/baligh_final_qwen3_1.7b/train.jsonl",
        "validation": "data/baligh_final_qwen3_1.7b/validation.jsonl",
    },
)
```


## 3. سكربت QLoRA كامل

احفظ الملف باسم:

```text
train_baligh_qwen3_17b.py
```

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "outputs/Baligh-1.7B"

set_seed(42)

bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
fp16 = torch.cuda.is_available() and not bf16

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/baligh_final_qwen3_1.7b/train.jsonl",
        "validation": "data/baligh_final_qwen3_1.7b/validation.jsonl",
    },
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name="baligh-qwen3-1.7b-sft",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_length=4096,
    packing=True,
    assistant_only_loss=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=bf16,
    fp16=fp16,
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },
    eos_token="<|im_end|>",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved adapter to: {OUTPUT_DIR}")
```

توصي وثائق TRL باستخدام conversational dataset مع `assistant_only_loss=True` حتى يُحسب الـ loss على رسائل المساعد فقط، بدل تدريب النموذج على نص المستخدم والنظام أيضًا.  كما أن Qwen3 معروف ضمن النماذج التي يمكن فيها استخدام chat template لإنتاج assistant masks، لكن يجب اختبار نسخة `transformers` و`trl` المثبتة عندك.[^9_2][^9_3]

## 4. التشغيل

ابدأ بتجربة صغيرة:

```bash
python train_baligh_qwen3_17b.py
```

إذا ظهر خطأ CUDA Out Of Memory، عدّل:

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
max_length = 2048
```

وإذا كان التدريب مستقرًا وذاكرة GPU متاحة، جرّب:

```python
max_length = 4096
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

لا تبدأ بـ 8192 أو أكثر قبل قياس طول العينات؛ طول السياق يرفع استهلاك الذاكرة بسرعة.

## 5. اختبار Baligh بعد التدريب

احفظ:

```python
# test_baligh.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "outputs/Baligh-1.7B"

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "أنت بليغ، مساعد عربي علمي ملتزم بمنهج أهل السنة والجماعة. "
            "لا تنسب قولًا بلا مصدر، وإذا لم يكف السياق فقل لا أعلم."
        ),
    },
    {
        "role": "user",
        "content": "ما الفرق بين النص الصريح والاستنباط الفقهي؟",
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )

answer = tokenizer.decode(
    output[^9_0][inputs["input_ids"].shape[^9_1]:],
    skip_special_tokens=True,
)

print(answer)
```

في الإنتاج، لا تعتمد على النموذج وحده في الأسئلة الدينية. اجعل Athar/RAG هو مصدر المعرفة، ودرّب Baligh على صياغة الإجابة، استخدام السياق، وذكر المصدر.

## 6. دمج الـ adapter

أثناء التطوير، اترك النموذج Adapter منفصلًا. قبل النشر يمكنك دمجه:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3-1.7B"
adapter = "outputs/Baligh-1.7B"
merged = "outputs/Baligh-1.7B-merged"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

model.save_pretrained(merged, safe_serialization=True)
tokenizer.save_pretrained(merged)
```

بعد ذلك اختبر النموذج المدموج مقابل:

```text
Qwen/Qwen3-1.7B
Baligh-1.5B
Baligh-1.7B adapter
Baligh-1.7B merged
```


## 7. إعداد البيانات قبل التدريب النهائي

قبل الضغط على التدريب النهائي:

- افصل `train` و`validation` حسب المصدر أو URL، وليس random split فقط.
- امنع تكرار السؤال أو الفتوى في الطرفين.
- لا تدخل `needs_review.jsonl` قبل المراجعة.
- راجع نسبة الأحاديث والمصادر.
- صنّف الإجابات إلى: عقيدة، فقه، حديث، تفسير، سيرة، لغة.
- احتفظ بالمذاهب عند وجودها.
- اختبر citations exact-match.
- أضف أمثلة «السياق غير كافٍ».
- أضف أمثلة ترفض اختلاق التخريج.
- أضف أمثلة تذكر الخلاف المعتبر.

من الأفضل البدء بـ 1–2 epoch فقط؛ فـ SFT الطويل على Dataset دينية صغيرة قد يسبب حفظًا زائدًا ونبرة جازمة أو تراجعًا في القدرات العامة. استخدم learning rate منخفضًا نسبيًا، ثم قارن checkpoint الأفضل حسب `eval_loss` وbenchmark سني مستقل، وليس loss وحده.

## 8. تقييم Baligh

أنشئ benchmark ثابتًا يتضمن:


| الفئة | ما الذي تقيسه |
| :-- | :-- |
| القرآن | رقم السورة والآية وصحة النص |
| الحديث | عدم اختلاق التخريج، وذكر الدرجة عند توفرها |
| العقيدة | الاتساق مع أصول أهل السنة |
| الفقه | ذكر الدليل والخلاف وعدم ادعاء الإجماع |
| RAG | الالتزام بالنص المسترجع |
| الرفض | التصريح بعدم كفاية المصدر |
| العربية | الفصحى، الوضوح، وعدم العامية غير المطلوبة |
| السلامة | عدم إعطاء فتوى شخصية جازمة دون تفاصيل |

الـ SFTTrainer يدعم conversational format وassistant-only loss، وهو مناسب لهذا النوع من بيانات Baligh.  ومشروع Baligh الحالي يحتوي على وحدات منفصلة للـ data loading والـ formatting والـ validation والـ training، لذلك يفضل نقل السكربت إلى بنية المشروع بعد التأكد من نجاح التجربة standalone.[^9_2]

## Artifact

لا يوجد ملف جديد؛ سكربت التدريب الكامل مضمّن في الرد باسم:

```text
train_baligh_qwen3_17b.py
```


## Next actions

1. شغّل smoke test على 1,000–5,000 عينة، وتأكد من عدم وجود أخطاء في `assistant_only_loss` أو chat template.
2. نفّذ تدريب QLoRA أولي بـ `max_length=2048` وepoch واحد، ثم قيّم citations والرفض والعقيدة.
3. بعد نجاح الاختبار، نفّذ التدريب النهائي على البيانات المراجعة فقط، ثم ارفع adapter وmodel card باسم `Baligh-1.7B`.
<span style="display:none">[^9_10][^9_11][^9_12][^9_13][^9_14][^9_15][^9_16][^9_4][^9_5][^9_6][^9_7][^9_8][^9_9]</span>

<div align="center">⁂</div>

[^9_1]: https://www.alibabacloud.com/help/en/pai/use-cases/deploy-fine-tune-and-evaluate-qwen3-in-quickstart/

[^9_2]: https://huggingface.co/docs/trl/v0.23.0/en/sft_trainer

[^9_3]: https://huggingface.tw/docs/trl/sft_trainer

[^9_4]: https://aws.amazon.com/blogs/machine-learning/improve-your-agents-tool-calling-accuracy-with-sft-and-dpo-on-amazon-sagemaker-ai/

[^9_5]: https://arxiv.org/html/2603.08640v2

[^9_6]: https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune

[^9_7]: https://huggingface.co/docs/openenv/en/tutorials/sft-warmup

[^9_8]: https://huggingface.co/lllyx/Qwen3-1.7B-SFT

[^9_9]: https://huggingface.co/datasets/Mustafaege/qwen3.5-toolcalling-v2

[^9_10]: https://nixiesearch.substack.com/p/fine-tuning-qwen3-at-home-to-respond

[^9_11]: https://discuss.huggingface.co/t/sfttrainerflags-blocks-assistant-only-loss-true/176210

[^9_12]: https://www.linkedin.com/pulse/fine-tuning-llm-your-domain-specific-responses-practical-vimal-singh-xmnec

[^9_13]: https://www.reddit.com/r/Qwen_AI/comments/1szqh1l/uncensored_qwen_36_27b_exceeds_all_quant_metrics/

[^9_14]: https://discuss.huggingface.co/t/finetuning-a-reasoning-llm-with-supervised-or-reinforcement-learning/176449

[^9_15]: https://teendifferent.substack.com/p/apply_chat_template-is-the-safety

[^9_16]: projects.islamic_llm.murad_sunni_schema


---

## 1. تثبيت المتطلبات

```bash
pip install -U \
  torch transformers datasets accelerate peft bitsandbytes \
  trl huggingface_hub
```

ثم سجّل الدخول:

```bash
hf auth login
```

استخدم أحدث إصدارات `transformers` و`trl`؛ لأن `assistant_only_loss` يعتمد على دعم الـ chat template لإرجاع assistant-token masks. توثيق TRL يوضح أن `assistant_only_loss=True` يعمل مع conversational datasets التي تحتوي على `messages`، ويحسب الخسارة على ردود المساعد فقط.[^10_1]

الملف الكامل الجاهز الذي أنشأته لك هو:

```text
train_baligh_qwen3_17b_complete.py
```

ويدعم التدريب، الدمج، والرفع.

## 2. ضبط assistant_only_loss

بياناتك يجب أن تكون بهذا الشكل:

```json
{
  "messages": [
    {"role": "system", "content": "أنت بليغ..."},
    {"role": "user", "content": "ما حكم ...؟"},
    {"role": "assistant", "content": "بحسب المصدر ..."}
  ]
}
```

الإعداد الأساسي:

```python
training_args = SFTConfig(
    assistant_only_loss=True,
    eos_token="<|im_end|>",
    max_length=2048,
    packing=False,
)
```

في TRL، هذا الخيار لا يعمل مع أي template عشوائي؛ يجب أن يستطيع الـ tokenizer إنتاج `assistant_masks`. لذلك اختبره قبل التدريب:

```python
messages = [
    {"role": "system", "content": "أنت بليغ."},
    {"role": "user", "content": "عرّف الحديث الصحيح."},
    {"role": "assistant", "content": "هو ما اتصل سنده..."},
]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_assistant_tokens_mask=True,
    return_tensors="pt",
)

print(encoded.keys())
print(encoded.get("assistant_masks"))
```

إذا كانت النتيجة `assistant_masks` كلها أصفار أو لم يكن المفتاح موجودًا، فلا تبدأ التدريب. غالبًا لديك إصدار قديم من `transformers/trl` أو template لا يدعم assistant generation markers. وثائق TRL تشترط دعم `{% generation %}` و`{% endgeneration %}` في قالب المحادثة لهذا الخيار.[^10_2][^10_1]

إذا واجهت مشكلة مع `assistant_only_loss=True`، استخدم صيغة prompt-completion:

```json
{
  "prompt": [
    {"role": "system", "content": "أنت بليغ..."},
    {"role": "user", "content": "ما حكم ...؟"}
  ],
  "completion": [
    {"role": "assistant", "content": "بحسب المصدر ..."}
  ]
}
```

ثم:

```python
SFTConfig(
    completion_only_loss=True,
    assistant_only_loss=False,
)
```

لكن مع Qwen3 وبيانات `messages`، الخيار الأول هو الأنسب.

## 3. إعداد LoRA للنماذج العربية

الإعداد الذي أبدأ به لـ Qwen3-1.7B:

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
```

استهداف طبقات attention وMLP معًا عادةً يعطي قدرة تكييف أعلى من استهداف `q_proj` و`v_proj` فقط، لكنه يستخدم ذاكرة ووقتًا أكثر.[^10_3]

### Presets

| الهدف | `r` | `alpha` | target modules |
| :-- | --: | --: | :-- |
| Smoke test | 16 | 32 | `q_proj`, `v_proj` |
| Baligh production baseline | 32 | 64 | Attention + MLP |
| Maximum adaptation | 64 | 128 | Attention + MLP |

ابدأ بـ `r=32`. لا ترفع إلى 64 إلا إذا وجدت أن النموذج لا يتعلم أسلوب Baligh أو لا يلتزم بالتنسيق المطلوب.

مهم: لا تجعل LoRA يتعلم حقائق دينية غير موثقة. يجب أن يتعلم أسلوب الإجابة، تنظيمها، إظهار المصدر، والامتناع عند نقص الدليل، بينما يبقى Athar/RAG مصدر المعرفة.

## 4. تشغيل التدريب

بعد تشغيل سكربت تجهيز البيانات:

```bash
python prepare_baligh_final_qwen3_17b.py \
  --output-dir data/baligh_final_qwen3_1.7b
```

ابدأ بتجربة صغيرة:

```bash
python train_baligh_qwen3_17b_complete.py train \
  --data data/baligh_final_qwen3_1.7b \
  --output outputs/baligh-smoke \
  --epochs 1 \
  --max-length 1024 \
  --batch-size 1 \
  --grad-accum 16 \
  --eval-steps 50
```

التدريب النهائي:

```bash
python train_baligh_qwen3_17b_complete.py train \
  --data data/baligh_final_qwen3_1.7b \
  --output outputs/Baligh-1.7B-adapter \
  --epochs 2 \
  --max-length 2048 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --eval-steps 250
```

لو لديك GPU بذاكرة أكبر:

```bash
python train_baligh_qwen3_17b_complete.py train \
  --data data/baligh_final_qwen3_1.7b \
  --output outputs/Baligh-1.7B-adapter \
  --epochs 2 \
  --max-length 4096 \
  --batch-size 2 \
  --grad-accum 8 \
  --packing
```

Unsloth يدعم fine-tuning لـ Qwen3 مع 4-bit training، ويوفر workflow مناسبًا للنماذج الصغيرة، لكن لا تستخدم GGUF للتدريب؛ استخدم checkpoint بصيغة Transformers أو نسخة 4-bit مناسبة للتدريب.[^10_4]

## 5. تقييم Baligh

لا تعتمد على `eval_loss` وحده. استخدم ثلاث طبقات.

### أ. تقييم آلي

تابع:

```text
eval_loss
mean_token_accuracy
gradient norm
عدد التوكنات
```

الـ `eval_loss` الأقل جيد، لكنه لا يقيس صحة الحديث أو التزام النموذج بالمنهج السني.

### ب. Benchmark ديني

أنشئ JSONL منفصلًا، مثل:

```json
{
  "id": "hadith_001",
  "category": "hadith_attribution",
  "question": "ما درجة الحديث المذكور؟",
  "context": "...",
  "expected_behavior": "لا يجزم دون مصدر تخريج",
  "required_citation": true
}
```

قسّم التقييم إلى:


| الفئة | المؤشر |
| :-- | :-- |
| القرآن | exact verse/reference accuracy |
| الحديث | attribution وعدم اختلاق التخريج |
| العقيدة | الالتزام بأصول أهل السنة |
| الفقه | ذكر الدليل والخلاف وعدم ادعاء الإجماع |
| RAG | answer faithfulness للسياق |
| الرفض | الاعتراف بعدم كفاية المصدر |
| العربية | الفصحى والوضوح |
| citations | صحة الكتاب والمؤلف والصفحة |

### ج. تقييم بشري

راجع على الأقل 300–500 سؤال، بدرجات من 0 إلى 4:

```text
source_faithfulness
citation_correctness
sunni_alignment
fiqh_precision
uncertainty_calibration
arabic_quality
harmfulness
```

اعمل split بالتنويع الموضوعي، وليس random فقط، حتى لا تظهر نفس الفتوى أو السؤال بصياغة مختلفة في التدريب والتقييم.

## 6. دمج LoRA مع Qwen3

بعد انتهاء التدريب، سيكون لديك Adapter فقط. ادمجه مع النموذج الأصلي:

```bash
python train_baligh_qwen3_17b_complete.py merge \
  --adapter outputs/Baligh-1.7B-adapter \
  --merged outputs/Baligh-1.7B-merged
```

العملية داخليًا:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/Baligh-1.7B-adapter",
)

merged_model = model.merge_and_unload(safe_merge=True)
merged_model.save_pretrained(
    "outputs/Baligh-1.7B-merged",
    safe_serialization=True,
)
```

`merge_and_unload()` يدمج تحديث LoRA داخل أوزان النموذج الأصلي، وبعدها يمكنك تشغيل النموذج بدون PEFT adapter منفصل.[^10_5]

احتفظ دائمًا بنسختين:

```text
Baligh-1.7B-adapter
Baligh-1.7B-merged
```

نسخة الـ adapter أصغر وأسهل لإعادة الاستخدام، والنسخة المدموجة أسهل في النشر.

## 7. رفع النموذج إلى Hugging Face

سجّل الدخول:

```bash
hf auth login
```

ثم ارفع النسخة المدموجة:

```bash
python train_baligh_qwen3_17b_complete.py upload \
  --folder outputs/Baligh-1.7B-merged \
  --repo Kandil7/Baligh-1.7B \
  --private
```

للنشر العام، احذف `--private`:

```bash
python train_baligh_qwen3_17b_complete.py upload \
  --folder outputs/Baligh-1.7B-merged \
  --repo Kandil7/Baligh-1.7B
```

الـ script يستخدم `create_repo()` ثم `upload_folder()` ويرفع ملفات `safetensors`, `config.json`, tokenizer، والـ documentation. Hugging Face توصي باستخدام `upload_folder()` للمجلدات، وهو يدعم الرفع القابل للاستئناف للملفات الكبيرة.[^10_6]

أضف `README.md` إلى مجلد النموذج قبل الرفع:

```markdown
---
base_model:
- Qwen/Qwen3-1.7B
language:
- ar
pipeline_tag: text-generation
library_name: transformers
license: apache-2.0
---

# Baligh-1.7B

Arabic Sunni-domain instruction-tuned model based on Qwen3-1.7B.

## Intended use

Arabic Islamic knowledge assistance with retrieval grounding.

## Important limitations

The model is not a mufti and must not replace qualified scholarly consultation.
Answers should be grounded in retrieved sources and checked for citation accuracy.

## Training

Fine-tuned with QLoRA and supervised fine-tuning.

## Evaluation

Report Arabic quality, citation correctness, source faithfulness, Sunni alignment,
and refusal behavior separately.
```


## إعدادات Qwen3 أثناء inference

Qwen3 يدعم نمط التفكير والنمط المباشر. أثناء تقييم Baligh، ثبّت الوضع حتى تكون المقارنة عادلة:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

ثم استخدم:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
)
```

توثيق Qwen3 يذكر أن `enable_thinking=False` يمنع إخراج كتلة `<think>...</think>`، كما يوصي بإعدادات مختلفة للنمط المباشر ونمط التفكير.[^10_4]

## تنبيه على السكربت

في الملف الذي أنشأته، دالة `load_data()` تتوقع:

```text
data/baligh_final_qwen3_1.7b/train.jsonl
data/baligh_final_qwen3_1.7b/validation.jsonl
```

ويجب أن تحتوي البيانات على `messages`. كذلك لا تجعل `packing=True` في أول smoke test؛ اختبر `assistant_masks` أولًا، ثم فعّل packing بعد التأكد من صحة الـ labels.

## Artifact

الملف البرمجي الكامل الجاهز: **`train_baligh_qwen3_17b_complete.py`**

## Next actions

1. اختبر `assistant_masks` على عينة واحدة، ثم شغّل smoke test بـ 1,000 عينة و`max_length=1024`.
2. قيّم checkpoint على benchmark سني يحتوي على citations، العقيدة، الفقه، الحديث، والرفض عند نقص السياق.
3. ادمج الـ adapter بـ `safe_merge=True`، ارفع النسخة المدموجة إلى `Kandil7/Baligh-1.7B`، وأضف model card يوضح المصادر والتراخيص والقيود.
<span style="display:none">[^10_10][^10_11][^10_12][^10_13][^10_14][^10_15][^10_16][^10_17][^10_7][^10_8][^10_9]</span>

<div align="center">⁂</div>

[^10_1]: https://huggingface.co/docs/trl/v0.23.0/en/sft_trainer

[^10_2]: https://discuss.huggingface.co/t/sft-conversation-llama3-8b-instruct-fails-with-assistant-only-loss-true/173101

[^10_3]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

[^10_4]: https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune

[^10_5]: https://www.spheron.network/blog/fine-tune-moe-llms-gpu-cloud-expert-parallelism-load-balancing-moelora-guide/

[^10_6]: https://huggingface.co/docs/huggingface_hub/en/guides/upload

[^10_7]: https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb

[^10_8]: https://huggingface.co/docs/diffusers/en/using-diffusers/push_to_hub

[^10_9]: https://www.gilesthomas.com/2026/01/custom-automodelforcausallm-frompretrained-models-on-hugging-face

[^10_10]: https://www.datacamp.com/cheat-sheet/hugging-face-cheat-sheet

[^10_11]: https://forums.developer.nvidia.com/t/fine-tuning-qwen-qwen3-vl-30b-a3b-instruct-fp8-with-qlora-on-dgx-spark/351329

[^10_12]: https://www.linkedin.com/pulse/ai-parallel-training-walkthrough-joshua-reuben-gxfzf

[^10_13]: https://discuss.huggingface.co/t/deprecation-of-assistant-only-loss/175041

[^10_14]: https://www.marktechpost.com/2026/06/02/how-to-fine-tune-lfm2-using-qlora-and-dpo-a-complete-step-by-step-coding-tutorial-on-google-colab/

[^10_15]: https://www.linkedin.com/pulse/finding-your-way-around-hugging-face-practical-guide-oyerinde-gmcpn-oktuf

[^10_16]: https://discuss.huggingface.co/t/sfttrainerflags-blocks-assistant-only-loss-true/176210

[^10_17]: https://huggingface.tw/docs/trl/sft_trainer

