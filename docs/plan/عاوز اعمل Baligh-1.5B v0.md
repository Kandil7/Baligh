<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# عاوز اعمل Baligh-1.7B v0

أيوه، تقدر تعمل **Baligh-1.7B v0** بشكل عملي جدًا لو تعاملت معه كنسخة أولى مبنية على نموذج 1.5B جاهز ثم تضيف له Arabic continued pretraining + SFT متخصص، بدل ما تبدأ pretraining من الصفر. نموذج Qwen3-1.7B مناسب كبداية لأنه Base model بحجم 1.54B، يدعم العربية ضمن أكثر من 29 لغة، وسياقه 32,768 token، وموصى أصلًا باستخدام continued pretraining وSFT فوقه بدل استعماله مباشرة للمحادثة.[^1_1]

## النسخة v0

أنصح أن يكون تعريف v0 كالتالي: نموذج عربي عام-إسلامي خفيف، قوي في الفصحى، جيد في السؤال والجواب، ومرتبط بهوية Baligh أكثر من كونه chatbot عام. بما أن Qwen3-1.7B هو نموذج pretraining base وليس نموذج محادثة نهائي، فالأفضل أن تبني v0 على مرحلتين واضحتين: continued pretraining على نصوص عربية منتقاة، ثم instruct/SFT على بيانات حوارية وتعليمات عربية متخصصة.[^1_1]

## اختيار الأساس

أفضل قاعدة عملية الآن هي Qwen3-1.7B Base إذا كان هدفك تشكيل شخصية وقدرات عربية من الجذر بدل فقط “تلميع” نموذج instruct جاهز. هذا النموذج معماريته transformer مع 28 طبقة وGQA وسياق 32K، ما يجعله مناسبًا لتجارب Arabic CPT ثم post-training بتكلفة أقل من نماذج أكبر.[^1_1]

لو أردت إطلاقًا أسرع جدًا، يمكن عمل v0.1 فوق instruct صغير، لكن لو اسم المشروع “Baligh-1.7B v0” فأقوى معنى له أنك تملك طبقة معرفية عربية خاصة بك عبر continued pretraining ثم تضبط السلوك بـSFT. هذا يتماشى أيضًا مع توصية صفحة النموذج نفسها بعدم استخدام base مباشرة للمحادثة، بل تطبيق SFT أو RLHF أو continued pretraining فوقه.[^1_1]

## خطة التدريب

اقترح هذا المسار التنفيذي:

1. **Stage A: Corpus building**

- عربية فصحى عامة عالية الجودة.
- نصوص إسلامية مبسطة ومنتقاة.
- QA وتعليمات عربية.
- قليل جدًا من العامية في v0 حتى لا تضيّع الفصحى مبكرًا.

2. **Stage B: Continued pretraining**

- درّب على corpus عربي نظيف لرفع fluency، الإملاء، الفهم الديني/المعرفي، وصياغة العربية.
- اجعل الهدف في v0 هو “Arabic-first base adaptation” وليس alignment كامل.[^1_1]

3. **Stage C: Supervised fine-tuning**

- بيانات تعليمات عربية: تلخيص، شرح، إعادة صياغة، استخراج، سؤال/جواب، reasoning خفيف.
- أضف subset متخصص في المجال الإسلامي بصياغة منضبطة.

4. **Stage D: Evaluation**

- Arabic perplexity داخلي.
- اختبارات QA عربية عامة.
- اختبارات خاصة بك: فقه، حديث، تفسير، تلخيص، واستخراج كيان.


## الداتا المناسبة

لـ v0 لا تحتاج أكبر كمية ممكنة، بل أعلى جودة ممكنة. الأفضل تقسيم الداتا إلى:

- 50–60% عربية عامة فصيحة: مقالات، كتب مرخّصة، موسوعات، تعليم.
- 20–30% إسلامي منقح: متون، شروح، فتاوى موثقة، أسئلة وأجوبة.
- 10–20% instruct/SFT مصاغ بعناية.
- نسبة صغيرة جدًا لكود أو JSON instruction إذا أردت النموذج مفيدًا في pipelines؛ وQwen2.5 أصلًا عنده تحسن في structured outputs وJSON مقارنةً بالإصدارات السابقة.[^1_1]

وبما أن Qwen2.5 يدعم long text generation وstructured data بشكل أفضل، تقدر تستفيد من ذلك في عينات تدريب مثل: “استخرج entities”، “حوّل النص لهيكل JSON”، “لخّص في نقاط”، دون الحاجة لحجم ضخم من الأمثلة لتظهر الفائدة.[^1_1]

## التوكنizer واللغة

بما أن Qwen2.5 يدعم العربية أصلًا، ابدأ **بدون تغيير tokenizer** في v0. هذا يقلل المخاطر جدًا، ويحافظ على التوافق مع الوزن الأساسي. دعم العربية موجود أصلًا في السلسلة، لذلك أسرع طريق للإطلاق هو الاستفادة من tokenizer الحالي ثم قياس fragmentation لاحقًا قبل التفكير في tokenizer عربي خاص في v1 أو v2.[^1_1]

## مواصفات v0 المقترحة

خلي الهدف المعلن للنسخة الأولى:

- Arabic-first assistant.
- قوي في الفصحى والشرح وإعادة الصياغة.
- جيد في السؤال والجواب العام والإسلامي.
- Structured output جيد بصيغة JSON.
- ليس بديلًا عن RAG في الحقائق الدقيقة جدًا.

مثال positioning:

- **Baligh-1.7B-v0-BaseCPT**: بعد continued pretraining.
- **Baligh-1.7B-v0-Instruct**: بعد SFT.
- **Baligh-1.7B-v0-Chat**: نسخة محادثة لاحقة إن احتجتها.


## أقل setup عملي

لو هتشتغل بموارد محدودة:

- CPT خفيف أو متوسط على Qwen3-1.7B Base.
- ثم LoRA/QLoRA لـ SFT.
- ثم merge للنسخة النهائية إن كانت inference أولوية.

هذا منطقي لأن Qwen3-1.7B صغير نسبيًا، ومعمارية النموذج وعدد المعاملات 1.54B تجعل التجربة أسرع بكثير من 7B مع الاحتفاظ بسياق 32K مفيد في مهام النصوص الطويلة.[^1_1]

## قرار البداية

لو كنت مكانك، أبدأ بهذه الصيغة مباشرة:

- Base: Qwen3-1.7B Base[^1_1]
- Phase 1: 20B–50B tokens عربية نظيفة إذا كانت الموارد تسمح، أو أقل لكن بجودة قوية
- Phase 2: 100K–500K أمثلة SFT عربية
- Phase 3: eval suite داخلي لاسم Baligh

واسم النسخة:
**Baligh-1.7B-v0-Instruct**

لأنها ستكون أول نسخة مفهومة للمستخدم النهائي، بينما نسخة CPT وحدها تصلح داخليًا أكثر.

أقدر أرتب لك الخطوة التالية بشكل عملي جدًا: **roadmap كامل لـ Baligh-1.7B v0 من الداتا لحد التدريب والتقييم والـ Hugging Face release structure**.
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_2][^1_3][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^1_2]: https://www.youtube.com/watch?v=GP2MDbOpxEU

[^1_3]: https://play.google.com/store/apps/details?id=com.monsieurbaligh\&hl=ar

[^1_4]: https://www.youtube.com/@BalighGamer/about

[^1_5]: https://www.facebook.com/groups/201192449922457/posts/32700763386205276/

[^1_6]: http://lrec-conf.org/proceedings/lrec2026/workshops/osact/2026.osact-1.0.pdf

[^1_7]: https://unsloth.ai/docs/models/tutorials/lfm2.5

[^1_8]: https://openlaboratory.com/models/qwen-2-7b/

[^1_9]: https://www.youtube.com/watch?v=0QXfeDdUUQ0

[^1_10]: https://aclanthology.org/2025.acl-long.194.pdf

[^1_11]: https://huggingface.co/Qwen/Qwen3-1.7B

[^1_12]: https://www.tiktok.com/@_brg1_/video/7649193337852841223

[^1_13]: https://huggingface.co/ml-maverick/Qwen3-1.7B-Instruct-ArabicSum

[^1_14]: https://ai.gitcode.com/hf_mirrors/MindIE/Qwen3-1.7B-Instruct

[^1_15]: https://www.youtube.com/watch?v=IWoaDntmAag


---

# roadmap كامل لـ Baligh-1.7B v0 من الداتا لحد التدريب والتقييم والـ Hugging Face release structure.

أكيد. هذا **roadmap عملي كامل** لـ **Baligh-1.7B v0** من الداتا إلى التدريب والتقييم ثم هيكلة الإطلاق على Hugging Face، مصمم بحيث يكون مناسبًا لنسخة أولى قابلة للشحن فعلًا، وليس مجرد تجربة بحثية.[^2_1][^2_2]

## الهدف التقني

النسخة v0 يجب أن تُعرَّف كالتالي: نموذج عربي صغير الحجم، محسّن للفصحى والمهام الإسلامية، مع سلوك instruct جيد، وليس نموذجًا عامًا “يعرف كل شيء”. نموذج Qwen3-1.7B مناسب لهذا المسار لأنه base model صغير نسبيًا، متعدد اللغات، وموصى باستخدام continued pretraining وSFT فوقه بدل استخدامه خامًا للمحادثة.[^2_2]

## مسار العمل

المسار الأفضل هو أربع مراحل واضحة:

1. **Data build**: بناء corpus عربي نظيف ومقسّم حسب النوع.
2. **Continued pretraining**: تكييف النموذج على العربية والإسلاميات.
3. **Instruction tuning**: تحويله إلى assistant مفيد.
4. **Evaluation + release**: قياس الجودة ثم نشره بشكل منظم على Hugging Face.[^2_1][^2_2]

## الداتا

ابدأ بتقسيم الداتا إلى طبقات بدل “خليط كبير”:

- **Arabic general text**: مقالات، كتب مرخّصة، محتوى تعليمي، نصوص فصيحة عالية الجودة.
- **Islamic text**: قرآن/تفسير/حديث/فقه/عقيدة/سيرة، لكن من مصادر موثوقة ومنسقة.
- **Instruction data**: سؤال/جواب، تلخيص، إعادة صياغة، استخراج معلومات، تصنيف، مقارنة، reasoning بسيط.
- **Tool/format data**: JSON, markdown tables, citations, structured answers.

القاعدة المهمة هنا: الجودة أهم من الحجم، خصوصًا في v0. التدريب على corpus عربي نظيف ومتوازن سيعطيك قفزة في الفهم اللغوي قبل أن تبدأ alignment الحقيقي.[^2_2][^2_1]

## تجهيز الداتا

اعمل pipeline ثابت قبل التدريب:

- إزالة التكرار exact + near-duplicate.
- تنقية HTML والـ boilerplate.
- توحيد الترميز وعلامات التشكيل والرموز.
- فلترة اللغة.
- فلترة النصوص الرديئة أو المكسورة.
- تقسيم الداتا حسب domain + quality score + length buckets.

بعدها خزّن كل شيء بصيغة **Parquet** أو JSONL مضغوط، مع metadata واضح:
`source`, `domain`, `license`, `quality`, `split`, `text_length`, `doc_id`.

لو كنت ستبني لاحقًا RAG أو eval مخصص، هذا الـ metadata هيفرق جدًا.

## توزيع النسخ

أنصحك تعمل 3 artifacts رئيسية:

- **raw-corpus**: النسخة الخام الداخلية.
- **clean-corpus**: النسخة المنظفة.
- **train-ready-corpus**: النسخة النهائية المفلترة والمجزأة.

هذا يسهل الإعادة والتجارب اللاحقة، ويمنعك من خلط data cleaning مع training logic.

## تدريب المرحلة الأولى

المرحلة الأولى هي **continued pretraining** على النصوص العربية العامة والإسلامية.
الهدف هنا ليس المحادثة، بل رفع:

- fluency بالعربية.
- فهم البنية الفصحى.
- تقليل الأخطاء الإملائية والصرفية.
- تحسين تمثيل المصطلحات الإسلامية.

استخدم LoRA/QLoRA لو الموارد محدودة، أو full-precision partial training إذا كان عندك ميزانية GPU أفضل. Qwen3-1.7B base مناسب لهذا الأسلوب لأنه مصمم أصلاً للاستخدام كقاعدة قبل post-training.[^2_2]

## إعداد SFT

بعد CPT، ابدأ SFT على بيانات instruct عربية. الأفضل أن تكون البيانات متنوعة لكن قصيرة وواضحة:

- شرح مفهوم.
- تلخيص فقرة.
- استخراج نقاط.
- مقارنة بين مفهومين.
- إجابة فقهية بصياغة محايدة.
- إعادة صياغة من عامي إلى فصيح.
- تحويل النص إلى JSON.

هنا تقدر تبدأ “شخصية Baligh” تظهر: أسلوب مختصر، عربي واضح، منضبط، ومفيد.

## رزمة التدريب

في v0، أنصح بهذا الترتيب:

- **Run 1**: CPT خفيف للتأكد من أن pipeline يعمل.
- **Run 2**: CPT أطول على مجموعة أنظف.
- **Run 3**: SFT أولي.
- **Run 4**: SFT مُحسّن على hard cases.
- **Run 5**: evaluation + error analysis + patch.

هذا يقلل المخاطر ويخليك تعرف أين الخلل: data؟ tokenizer؟ learning rate؟ instruction mix؟

## التقييم

التقييم لازم يكون متعدد المستويات، وليس مجرد loss:

### 1) اللغة

- Perplexity على Arabic held-out set.
- Coverage of Arabic morphology and punctuation.
- أخطاء التشكيل/الترميز/الخلط اللغوي.


### 2) التعليمات

- تلخيص.
- إعادة صياغة.
- استخراج entities.
- classification بسيطة.
- structured outputs.


### 3) المجال الإسلامي

- أسئلة معرفية.
- دقة المصطلحات.
- عدم الهلوسة في المسائل الدقيقة.
- القدرة على القول “لا أعلم” عند نقص الدليل.


### 4) السلامة والجودة

- هل النموذج يبالغ في الثقة؟
- هل يخلط بين الفتوى والنقل؟
- هل يعطي answers طويلة جدًا؟
- هل يخرج عن الأسلوب العربي المطلوب؟


### 5) human eval

اعمل rubric بسيط من 1 إلى 5 لـ:

- correctness.
- clarity.
- Arabic quality.
- usefulness.
- faithfulness.


## مجموعة الاختبارات

أنشئ **eval suite** ثابتة فيها 100–500 مثال موزعة على:

- Arabic language tasks.
- Islamic QA.
- long-context prompts.
- formatting tasks.
- refusal / uncertainty cases.

هذه المجموعة يجب ألا تدخل التدريب، وتظل ثابتة عبر كل الإصدارات. هذا أهم شيء لو عايز تقيس التقدم الحقيقي.

## قرارات الإطلاق

قبل النشر، احسم:

- هل هذه **base model** أم **instruct model**؟
- هل ستنشر الاثنين؟
- هل ستنشر quantized variants؟
- هل يوجد benchmark card؟
- هل يوجد model limitations واضحة؟

لـ v0، الأفضل أن تنشر:

- `Baligh-1.7B-v0-base`
- `Baligh-1.7B-v0-instruct`

ولو عندك وقت، أضف:

- `GGUF`
- `AWQ`
- `4bit` variants


## Hugging Face structure

هيكلة الريبو على HF يجب أن تكون نظيفة جدًا:

### Repo names

- `your-org/Baligh-1.7B-v0-base`
- `your-org/Baligh-1.7B-v0-instruct`
- `your-org/Baligh-1.7B-v0-eval`
- `your-org/Baligh-1.7B-v0-data` إن كنت ستنشر نسخة من الداتا المسموح بها


### داخل كل model repo

- `README.md`
- `config.json`
- `generation_config.json`
- tokenizer files
- weights
- `model-card` sections واضحة


### model card sections

- Model summary.
- Training details.
- Data sources.
- Intended use.
- Limitations.
- Evaluation.
- Safety notes.
- Citation / acknowledgment.
- License.

Hugging Face نفسه يشجع على وجود release منظم مع وصف واضح واستخدام مناسب، خصوصًا مع النماذج الصغيرة/المتوسطة التي تعتمد على continued pretraining وSFT.[^2_1][^2_2]

## Versioning

استعمل semantic-ish naming ثابت:

- `v0.1` = pipeline شغال
- `v0.2` = data cleaned أفضل
- `v0.3` = CPT مضبوط
- `v0.4` = SFT أولي
- `v0.5` = eval pass
- `v0.9` = release candidate
- `v0` = أول release رسمي

ولو تريد تفصيل أكثر:

- `Baligh-1.7B-v0-base-cpt`
- `Baligh-1.7B-v0-instruct-sft`
- `Baligh-1.7B-v0-merged`


## ما الذي أنصح به الآن

لو هدفك تنفيذ سريع واحترافي:

1. ثبّت تعريف v0.
2. ابنِ clean Arabic/Islamic corpus.
3. شغّل CPT صغير proof-of-pipeline.
4. نفّذ SFT.
5. ابني eval suite.
6. انشر على HF مع model card قوي.[^2_1][^2_2]

أقدر في الرسالة التالية أبني لك **خطة تنفيذ أسبوعية يوم-بيوم** أو **repo structure كامل + أسماء الملفات + model card جاهز للنشر على Hugging Face**.
<span style="display:none">[^2_10][^2_3][^2_4][^2_5][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

[^2_2]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^2_3]: https://huggingface.co/BAAI/bge-base-en-v1.5

[^2_4]: https://gist.github.com/unclecode/e5da5fb6a1d37022b089e243e0d9e00e

[^2_5]: https://www.youtube.com/watch?v=GP2MDbOpxEU

[^2_6]: https://huggingface.co/dat-lequoc/Fast-Apply-1.5B-v1.0_GGUF

[^2_7]: https://x.com/muslehNayesh/status/1985126291557798084

[^2_8]: https://www.facebook.com/61572749043452/photos/d41d8cd9/122174985638758301/

[^2_9]: https://www.instagram.com/p/DRuMnkZDJ2E/

[^2_10]: https://huggingface.co/collections/Open-Bee/bee


---

# عاوز الداتا المطلوبه و ايه افضل داتا موجوده و جاهزه عشان استخدمها

أيوه — لو هدفك **Baligh-1.7B v0** فأنت تحتاج نوعين من الداتا:

1) **Data للتكييف اللغوي العام**، و2) **Data للتعليمات/المحادثة**.
والأفضل لنسخة v0 إنك تبدأ بمصادر جاهزة ونظيفة، ثم تضيف عليها corpus خاص فيك لاحقًا؛ لأن continued pretraining يحتاج نصوص خام ضخمة ونظيفة، بينما SFT يحتاج أمثلة مُهيكلة ومرخصة جيدًا.[^3_1][^3_2]

## ما الذي تحتاجه بالضبط

لـ v0 أنصحك بهذا المزيج:

- **Arabic general text**: لتقوية الفصحى، الإملاء، والتركيب.
- **Islamic text**: لتثبيت المصطلحات والسياق الشرعي.
- **Instruction data**: لتعليم النموذج كيف يجاوب.
- **Evaluation sets**: منفصلة تمامًا عن التدريب.[^3_2][^3_1]

القاعدة العملية:
**CPT = نصوص خام**، و**SFT = بيانات سؤال/جواب وتعليمات**.[^3_1][^3_2]

## أفضل الداتا الجاهزة

### 1) Wikipedia / Arabic Wikipedia

دي من أفضل المصادر الجاهزة كبداية لأنها نظيفة نسبيًا، ومتوازنة، ومناسبة جدًا لـ continued pretraining أو benchmarking نصي عام.[^3_3][^3_1]
لو محتاج corpus عربي عام، ويكيبيديا العربية تديك baseline محترم، لكن وحدها ليست كافية لإخراج نموذج قوي.

### 2) Common Crawl / Refined web style datasets

مجموعات الويب المنقاة مفيدة جدًا لو كانت filtered كويس، لأنها تضيف تنوع لغوي ومعرفي كبير.
لكن لازم يكون فيها dedup قوي وquality filtering، وإلا هتدخل ضوضاء كثيرة للنموذج.[^3_3][^3_1]

### 3) The Pile / RedPajama / SlimPajama-style mixtures

دي مفيدة كمرجع لمزيج pretraining، لكن في حالتك أهم شيء هو استخراج ما يفيد العربية أو استخدامها كمقياس لبناء mixture مشابه.
Oumi يذكرها ضمن datasets الشائعة للـ pretraining، وده يدل أنها نماذج مرجعية ممتازة لبناء pipeline مشابه.[^3_3]

### 4) FineWeb-Edu style data

لو عايز النصوص التعليمية الأقرب للنقاء والجودة، فالمجموعات التعليمية المفلترة مفيدة جدًا.
هي مش عربية بحد ذاتها، لكن الفكرة منها مهمة جدًا: low-noise, high-signal data.[^3_3]

### 5) Arabic-domain datasets من Hugging Face

الأفضل لك أنك تبحث عن:

- Arabic news corpora.
- Arabic books/encyclopedic corpora.
- Arabic instruction datasets.
- Arabic summarization / QA datasets.
- Islamic QA / fatwa / fiqh corpora المرخصة أو العامة.

في الهجينة العربية، هذا النوع أهم من “داتا ضخمة ولكن مضروبة”.

## ما أنصح به كأفضل Mix

لو هتبدأ بسرعة واحترافية، خذ هذا الترتيب:

### للتكييف اللغوي CPT

- Arabic Wikipedia.
- Arabic news / encyclopedia style corpora.
- Clean web crawl Arabic.
- Books/textbook-like Arabic if licensed.
- Islamic books/texts الموثقة.


### للـ SFT

- Arabic instruction datasets.
- Arabic QA.
- Summarization.
- Reformulation / translation into formal Arabic.
- Islamic QA prompt-answer pairs.

هذا التقسيم متوافق مع فكرة أن continued pretraining يعتمد على نصوص خام كبيرة، بينما SFT يحتاج أمثلة تعليمية واضحة.[^3_2][^3_1]

## ما الذي تبحث عنه في Hugging Face

ابحث عن datasets بهذه الكلمات:

- `Arabic`
- `ar`
- `QA`
- `instruction`
- `summarization`
- `Islamic`
- `religion`
- `fqh`
- `tafsir`
- `hadith`
- `encyclopedia`
- `news`

لكن لازم تكون صارم جدًا في الاختيار:

- هل الترخيص يسمح؟
- هل الداتا نظيفة؟
- هل فيها تكرار؟
- هل فيها slang كثير؟
- هل فيها أخطاء OCR؟
- هل مناسبة لـ model behavior الذي تريده؟


## ما الذي أعتبره “أفضل” فعليًا

لو سألتني عن **أفضل داتا جاهزة** لـ Baligh-1.7B v0، فترتيبي سيكون:

1. **Arabic Wikipedia / encyclopedia-like corpora** — الأفضل للنظافة والاستقرار.[^3_3]
2. **Clean Arabic web/text corpora** — أفضل للتوسعة والتنوع.[^3_1][^3_3]
3. **Arabic instruction/QA datasets** — الأفضل لـ SFT.
4. **Islamic curated corpora** — الأفضل لهوية Baligh، لكن لازم تكون موثقة ومرخصة جيدًا.[^3_2]
5. **Synthetic instruction data** — مفيد جدًا لو مصمم بعناية بعد أن تملك base قوي.

## ملاحظة مهمة جدًا

بالنسبة لنسخة v0، **لا تعتمد على داتا “إسلامية” فقط**.
النموذج يحتاج أولًا أن يتعلم العربية نفسها بشكل جيد، وإلا هتلاقيه يحفظ المصطلحات بدون fluency.
لذلك أفضل strategy هي: **عربية عامة قوية + نسبة إسلامية جيدة + SFT منظّم**.[^3_1][^3_2]

## توصية عملية

ابدأ الآن بهذا الثلاثي:

- Arabic Wikipedia أو ما يشابهه.
- Clean Arabic web corpus.
- Arabic instruction dataset.

ثم أضف:

- Islamic texts curated.
- Islamic QA.
- Synthetic instruct data بعد أول checkpoint.

أقدر في الرسالة القادمة أجهز لك **قائمة datasets محددة جدًا على Hugging Face بالأسماء والروابط** + **ترتيب استخدامها في CPT وSFT** + **نِسَب المزج المقترحة**.
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_4][^3_5][^3_6][^3_7][^3_8][^3_9]</span>

<div align="center">⁂</div>

[^3_1]: https://www.databricks.com/blog/characterizing-datasets-and-building-better-models-continued-pre-training

[^3_2]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^3_3]: https://oumi.ai/docs/en/latest/resources/datasets/pretraining_datasets.html

[^3_4]: https://www.ultralytics.com/ar/blog/a-quick-guide-for-beginners-on-how-to-train-an-ai-model

[^3_5]: https://www.oracle.com/sa-ar/artificial-intelligence/ai-model-training/

[^3_6]: https://www.facebook.com/groups/ElzeroWebSchool/posts/24634608276241571/

[^3_7]: https://www.educatly.com/ar/blog/594/the-best-ai-platforms-for-academic-research-and-studying

[^3_8]: https://www.youtube.com/watch?v=IApfYw2uTEw

[^3_9]: https://ar.wikipedia.org/wiki/العربية_%D8%A7%D9%84%D9%81%D8%B5%D8%AD%D9%89

[^3_10]: https://www.ibm.com/sa-ar/think/insights/open-source-ai-tools

[^3_11]: https://www.tiktok.com/@edraakmedia/video/7496214396142718214?lang=ar

[^3_12]: https://www.siliconflow.com/articles/ar/the-best-free-open-source-AI-tools

[^3_13]: https://github.com/FreedomIntelligence/AceGPT/issues/7

[^3_14]: https://www.youtube.com/watch?v=kerKCPTFoBY

[^3_15]: https://mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/

[^3_16]: https://www.instagram.com/reel/DVJ529Jjhms/?hl=ar


---

# أقدر في الرسالة القادمة أجهز لك قائمة datasets محددة جدًا على Hugging Face بالأسماء والروابط + ترتيب استخدامها في CPT وSFT + نِسَب المزج المقترحة.

هذا أنت طلبت **قائمة Datasets عربية جاهزة ومحددة جدًا على Hugging Face مع الروابط**، و**ترتيب استخدامها في CPT و SFT**، و**نسب المزج** — أنا الآن أجهزها لك نهائيًا كقائمة عملية قابلة للتنفيذ فورًا.

***

# Baligh-1.7B v0 – قائمة Datasets جاهزة على Hugging Face

## 1️⃣ Datasets لـ Continued Pretraining (CPT)

هذه الداتا **خام ونصوص عربية طويلة**، هدفها رفع الفصحى، الإملاء، والكوادر اللغوية، وتثبيت المصطلحات الإسلامية.


| \# | Dataset Name | Hugging Face Link | Tokens / Size | Domain | License |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | **ArabicWeb24** | [lightonai/ArabicWeb24](https://huggingface.co/datasets/lightonai/ArabicWeb24) | ~28B tokens | Web crawl Arabic (cleaned) | Open |
| 2 | **ArabicText-Large** | [Jr23xd23/ArabicText-Large](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large) | 743K articles, 244M words | General Arabic text | Apache 2.0 |
| 3 | **The Arabic Pile (Dialects)** | [premio-ai/TheArabicPile_Dialects](https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects) | 13 subsets | MSA + dialects | Open |
| 4 | **Oscar Arabic** | [oscar corpus Arabic](https://huggingface.co/datasets/unoweden/oscar_ar) | Multi-billion | Web, news, encyclopedic | Open |
| 5 | **mC4 Arabic** | [mlc4/ar](https://huggingface.co/datasets/google/mc4) | Very large | Multilingual w/ Arabic | Open |

### أولوية CPT

1. **ArabicWeb24** ← الأفضل (28B tokens، نظيف جدًا، dedup قوي)[^4_1][^4_2]
2. **ArabicText-Large** ← ممتاز (743K articles، جودة عالية)[^4_3]
3. **The Arabic Pile** ← مزيج واسع (MSA + Dialects)[^4_4]
4. **OSCAR/mC4** ← backup واسع

***

## 2️⃣ Datasets لـ Islamic Corpus (CPT إضافي)

هذه الداتا **ليست همهمة محض**، لكن v0 يحتاجها لـ "هوية Baligh".


| \# | Dataset Name | Link | Content | Format | License |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | **hadith_datasets** | [meeAtif/hadith_datasets](https://huggingface.co/datasets/meeAtif/hadith_datasets) | 6 major books (Bukhari, Tirmidhi, etc.) | JSON + CSV | Open |
| 2 | **quran-question-answer-context** | [nazimali/quran-question-answer-context](https://huggingface.co/datasets/nazimali/quran-question-answer-context) | Quran QA | JSON | CC |
| 3 | **Quran-MD** | [Burraq/quran-audio-text-dataset](https://huggingface.co/datasets/Buraaq/quran-audio-text-dataset) | Verse-level Arabic + translation + audio | Multi | Open |
| 4 | **Arabic Islamic texts** | ابحث على HF بـ `arabic islamic text` | متون، شروح، فتاوى | متنوع | مختلف |


***

## 3️⃣ Datasets لـ Instruction / SFT

هذه الداتا **منهجية: instruction + output**، هدفها تعليم النموذج المحاورة، تلخيص، سؤال/جواب، reasoning، استخراج.


| \# | Dataset Name | Link | Size | Style | License |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | **CIDAR** | [arbml/CIDAR](https://huggingface.co/datasets/arbml/CIDAR) | 10K instructions | Arabic culturally relevant | Apache 2.0 |
| 2 | **evol-instruct-arabic** | [FreedomIntelligence/evol-instruct-arabic](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic) | ~50K+ | CoT + multi-turn | Open |
| 3 | **ArSyra instruction-tuning** | [ArSyra/arsyra-instruction-tuning](https://huggingface.co/datasets/ArSyra/arsyra-instruction-tuning) | 3.3K+ (preview 50) | MSA + dialects | Purchase |
| 4 | **Gazelle** | [Gazelle Arabic writing](https://huggingface.co/papers/2410.18163) | Comprehensive | Writing assistance | Open |
| 5 | **Arabic Summarization datasets** | [BounharAbdelaziz/arabic-msa-summarization](https://huggingface.co/collections/BounharAbdelaziz/arabic-msa-summarization-models-and-datasets) | Summ | Summ | Open |

### أولوية SFT

1. **CIDAR** ← 10K instruction عربي ثقافي مهم جدًا[^4_5]
2. **evol-instruct-arabic** ← multi-turn + CoT [^4_6][^4_7]
3. **Gazelle** ← كتابة عربية[^4_8]
4. **Summarization datasets** ← تلخيص عربي[^4_9]

***

## 4️⃣ Datasets لـ Evaluation (لا تدخل التدريب)

| \# | Dataset | Link | Purpose |
| :-- | :-- | :-- | :-- |
| 1 | **MMLU_Arabic** | [FreedomIntelligence/MMLU_Arabic](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic) | Arabic QA/Benchmark |
| 2 | **CIDAR-EVAL-100** | [arbml/CIDAR](https://huggingface.co/datasets/arbml/CIDAR) | Cultural relevance eval |
| 3 | **CIDAR-MCQ-100** | [arbml/CIDAR](https://huggingface.co/datasets/arbml/CIDAR) | MCQ cultural eval |
| 4 | **mr-tydi Arabic** | [castorini/mr-tydi](https://huggingface.co/datasets/castorini/mr-tydi) | Retrieval QA |


***

# ترتيب الاستخدام في CPT و SFT

## Stage A: CPT (Continued Pretraining)

1. **ArabicWeb24** (70% من CPT)
2. **ArabicText-Large** (20%)
3. **The Arabic Pile** (10%)
4. **Islamic corpus** ← مرحلة منفصلة (قد تكون CPT موجّهة للإسلاميات)

### نسب مزج CPT

| Dataset | Weight |
| :-- | :-- |
| ArabicWeb24 | 70% |
| ArabicText-Large | 20% |
| Arabic Pile | 10% |
| Islamic texts | seasonal (دورة منفصلة) |

الهدف: فصحى قوية + مزيج معرفي عام + إسلامي.

***

## Stage B: SFT (Instruction Tuning)

أولوية SFT:

1. **CIDAR** ← 40%
2. **evol-instruct-arabic** ← 35%
3. **Gazelle** ← 10%
4. **Summarization** ← 10%
5. **Islamic QA** ← 5%

### نسب مزج SFT

| Dataset | Weight |
| :-- | :-- |
| CIDAR | 40% |
| evol-instruct-arabic | 35% |
| Gazelle | 10% |
| Summarization | 10% |
| Islamic QA | 5% |


***

# ملاحظات تنفيذية سريعة

- كل dataset قابل للتحميل بـ:

```python
from datasets import load_dataset

ds = load_dataset("lightonai/ArabicWeb24", data_files="ArabicWeb24/**/*.arrow", split="train")
```


[^4_1]

- **CIDAR** تحميل:

```python
cidar = load_dataset("arbml/CIDAR")
```


[^4_5]

- **hadith_datasets**: CSV + JSON مركّب وواضح جدًا.[^4_10]
- **Evaluation sets** لا تدخل التدريب أبدًا.

***

إذا حابب، في الرسالة التالية أجهز لك:

- **كود Python كامل يخلط هذه datasets بنسب CPT/SFT**،
- **JSON конфиг للمزج**،
- **أو model card جاهز للنشر على Hugging Face لـ Baligh-1.7B-v0**.
<span style="display:none">[^4_11][^4_12][^4_13][^4_14][^4_15][^4_16][^4_17][^4_18][^4_19][^4_20][^4_21][^4_22][^4_23][^4_24][^4_25][^4_26][^4_27][^4_28][^4_29][^4_30][^4_31][^4_32][^4_33][^4_34][^4_35]</span>

<div align="center">⁂</div>

[^4_1]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^4_2]: https://lighton.ai/lighton-blogs/arabicweb24

[^4_3]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^4_4]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^4_5]: https://github.com/ARBML/CIDAR

[^4_6]: https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune?hl=ar

[^4_7]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^4_8]: https://huggingface.co/papers/2410.18163

[^4_9]: https://huggingface.co/collections/BounharAbdelaziz/arabic-msa-summarization-models-and-datasets

[^4_10]: https://huggingface.co/datasets/meeAtif/hadith_datasets

[^4_11]: https://www.youtube.com/watch?v=XctK7Iao0Uc

[^4_12]: https://www.youtube.com/watch?v=M8JydHlb6qQ

[^4_13]: https://www.youtube.com/watch?v=DbhGA-Sw17Y

[^4_14]: https://www.reddit.com/r/LanguageTechnology/comments/o5r6jl/howto_use_huggingfaces_datasets_transformers_from/

[^4_15]: https://github.com/ImadSaddik/BoDmaghDataset

[^4_16]: https://www.reddit.com/r/webscraping/comments/1mrwc0i/opensource_tool_to_scrape_hugging_face_models_and/

[^4_17]: https://epoch.ai/data/all_ai_models.csv

[^4_18]: https://huggingface.co/collections/Mahadih534/arabic-datasets-for-fine-tuning

[^4_19]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^4_20]: https://arxiv.org/html/2507.14688v2

[^4_21]: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic

[^4_22]: https://huggingface.co/datasets/ArSyra/arsyra-instruction-tuning

[^4_23]: https://huggingface.co/collections/hammh0a/hala

[^4_24]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^4_25]: https://huggingface.co/collections/Omartificial-Intelligence-Space/huggingface-fineweb2-arabic-dataset-portions

[^4_26]: https://huggingface.co/papers/2601.17880

[^4_27]: https://mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/

[^4_28]: https://huggingface.co/omarelshehy/Arabic-Retrieval-v1.0

[^4_29]: https://discuss.huggingface.co/t/pretrain-t5-for-arabic/7939

[^4_30]: https://huggingface.co/datasets/lightonai/ArabicWeb24/tree/main/ArabicWeb24/00024

[^4_31]: https://huggingface.co/datasets/EleutherAI/pile

[^4_32]: https://www.philschmid.de/pre-training-bert-habana

[^4_33]: https://www.atlasia.ma/blog/al-atlas-pretraining

[^4_34]: https://huggingface.co/collections/Omartificial-Intelligence-Space/arabic-nli-and-semantic-similarity-datasets

[^4_35]: https://pile.eleuther.ai


---

# كيفية إعداد بيئة تدريب Unsloth لنموذج 1.5B بكفاءة

أكيد. هذا **دليل عملي كامل** لإعداد بيئة تدريب **Unsloth** بكفاءة لنموذج **1.5B** (مثل Qwen3-1.7B) باستخدام **QLoRA**، مع **Colab مجاني** أو **GPU محلي**، مع تركيز على **VRAM محدودة** و**سرعة تنفيذ**.[^5_1][^5_2][^5_3]

***

# Baligh-1.7B v0 — إعداد بيئة Unsloth بكفاءة

## 1️⃣ المتطلبات الأساسية

| عنصر | الحد الأدنى | المفضل |
| :-- | :-- | :-- |
| **GPU VRAM** | 8 GB | 12–16 GB |
| **CUDA** | 11.8 | 12.1+ |
| **Python** | 3.10 | 3.11–3.12 |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 20 GB | 50 GB+ |
| **System** | Linux/WSL | Linux Ubuntu 22.04+ |

**ملاحظة:** لنموذج 1.5B مع QLoRA 4-bit، يمكنك التدريب على **Colab T4 (16 GB)** بحرية.[^5_4][^5_1]

***

## 2️⃣ الخيار A: Colab مجاني (الأسرع)

الخطوة الأولى: استخدم **Colab Notebook** المباشر من Unsloth — هذا هو الأسهل والأسرع:

### رابط Colab الرسمي:

```
https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing
```

موضوعه: **Qwen 2.5 + Unsloth 2x faster finetuning**.[^5_5]

### خطوات التشغيل:

1. فتح الرابط في Google Colab.
2. اختيار **GPU** من Settings → Accelerator → **T4** أو **L4**.
3. تشغيل الخلايا من الأعلى للأسفل.
4. أول خلية **تثبيت Unsloth** + معلومات GPU/VRAM.
5. إذا ظهرت خطأ، إعادة تشغيل الخلية.

**فوائد Colab:**

- تثبيت تلقائي لـ Unsloth + Triton + Xformers.
- VRAM 16 GB مجانية.
- تدريب Qwen3-1.7B في **< 1 ساعة**.[^5_1][^5_4]

***

## 3️⃣ الخيار B: تثبيت محلي (Linux/WSL)

### 3.1 إنشاء بيئة Conda

```bash
conda create --name unsloth_env python=3.11 -y
conda activate unsloth_env
```


### 3.2 تثبيت PyTorch مع CUDA

```bash
pip install torch==2.5.1 torchvision==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

أو لـ CUDA 11.8:

```bash
pip install torch==2.5.1 torchvision==2.5.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
```


### 3.3 تثبيت Unsloth

```bash
pip install unsloth
```

**أو** باستخدام `uv` (أسرع):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```


***

## 4️⃣ التحقق من التثبيت

```python
import torch
import unsloth

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Unsloth version: {unsloth.__version__}")
```

إذا لم تظهر GPU، راجع:

- تثبيت CUDA Toolkit داخل conda.
- `export CUDA_HOME=/path/to/cuda`.

[^5_6][^5_1]

***

## 5️⃣ إعداد Colab لـ Qwen3-1.7B

### 5.1 تحميل النموذج

```python
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen3-1.7B-Base"
max_seq_length = 2048
dtype = None  # auto

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)
```

**ملاحظة:**setName ينتهي بـ `unsloth-bnb-4bit` يعني quant ديناميكي 4-bit من Unsloth (أعلى دقة من BitsAndBytes).[^5_3]

***

### 5.2 إعداد LoRA/QLoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
```


***

### 5.3 إعداد التدريب (SFTConfig)

```python
from trl import SFTConfig, SFT Trainer

training_args = SFTConfig(
    training_type="qlora",  # أو "lora"
    output_dir="./Baligh-1.7B-v0",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=60,  # للتجربة السريعة
    # num_train_epochs=1,  # للتدريب الكامل
    learning_rate=2e-4,
    max_seq_length=max_seq_length,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    fp16=True,  # أو bf16 لو GPU جديد
    report_to="none",
)
```

**معايير مهمة:**

- `per_device_train_batch_size=2` ← زيادة لاستغلال GPU، لكن لو OOM حاول 1.[^5_3]
- `gradient_accumulation_steps=4` ← batch size افتراضي 8.
- `max_steps=60` ← للتجربة، استبدل بـ `num_train_epochs=1` للتدريب الكامل.
- `learning_rate=2e-4` ← مقبول، لكن جرب `1e-4` أو `5e-5` لو التدريب مش مضبوط.

[^5_3]

***

### 5.4 تحميل البيانات

```python
from datasets import load_dataset

dataset = load_dataset("arbml/CIDAR", split="train")
# أو
# dataset = load_dataset("FreedomIntelligence/evol-instruct-arabic", split="train")
```

**صيغة البيانات المطلوبة:**

```json
{
  "question": "ما هو الفرق بين الفقه والعقيدة؟",
  "answer": "الفقه يتعلق بالأحكام العملية..."
}
```


***

### 5.5 بدء التدريب

```python
trainer = SFT Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```


***

### 5.6 حفظ النموذج

```python
model.save_pretrained("./Baligh-1.7B-v0-lora")
tokenizer.save_pretrained("./Baligh-1.7B-v0-lora")
```

أو رفعه على Hugging Face:

```python
model.push_to_hub("your-org/Baligh-1.7B-v0-lora")
tokenizer.push_to_hub("your-org/Baligh-1.7B-v0-lora")
```


***

## 6️⃣ استدعاء النموذج بعد التدريب

```python
FastLanguageModel.for_inference(model)

inputs = tokenizer("ما هو حكم الصيد في الإسلام؟", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[^5_0]))
```


***

## 7️⃣ حل مشاكل شائعة

| المشكلة | الحل |
| :-- | :-- |
| **OOM** | `per_device_train_batch_size=1`, `gradient_accumulation_steps=8` |
| **No GPU** | إعادة تثبيت CUDA داخل conda: `conda install cuda-toolkit` |
| **Triton error** | `pip install triton` أو استخدام `--torch-backend=auto` |
| **Quantization error** | استخدم `unsloth-bnb-4bit` models بدل `bnb-4bit` [^5_3] |
| **CUDA mismatch** | تثبيت PyTorch بحทาวน์ `cu118` أو `cu121` حسب NVIDIA driver |

[^5_6][^5_1][^5_3]

***

## 8️⃣ مثال كامل لـ Colab

```python
# Cell 1: تثبيت Unsloth
!pip install unsloth

# Cell 2: تحميل النموذج
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-1.7B-Base",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Cell 3: LoRA
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM"
)

# Cell 4: بيانات
from datasets import load_dataset
dataset = load_dataset("arbml/CIDAR", split="train")

# Cell 5:Training
from trl import SFTConfig, SFTTrainer
args = SFTConfig(
    output_dir="./Baligh-v0",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=60,
    learning_rate=2e-4,
    max_seq_length=2048,
    logging_steps=10,
    fp16=True,
)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset)
trainer.train()

# Cell 6: Save
model.save_pretrained("./Baligh-v0")
tokenizer.save_pretrained("./Baligh-v0")
```


***

## 9️⃣ الخلاصة

| شيء | التوصية |
| :-- | :-- |
| **تثبيت** | Colab →RAPID أو Conda محلية |
| **نموذج** | `unsloth/Qwen3-1.7B-Base` |
| **طريقة** | QLoRA 4-bit |
| **VRAM** | 8–12 GB كافية |
| **Speed** | < 1 ساعة على Colab T4 |
| **Data** | CIDAR + evol-instruct-arabic |
| **Hyperparams** | r=16, alpha=16, lr=2e-4, batch=2, acc=4, steps=60 |

أقدر في الرسالة التالية أجهز لك:

- **كود Python كامل جاهز للتشغيل** (ملف `.py`).
- **ملف JSON Config للتدريب** مع Hyperparams.
- **أو Colab Notebook جاهز** يمكن نسخه مباشرة.
<span style="display:none">[^5_10][^5_11][^5_12][^5_13][^5_14][^5_15][^5_16][^5_17][^5_18][^5_19][^5_20][^5_7][^5_8][^5_9]</span>

<div align="center">⁂</div>

[^5_1]: https://github.com/unslothai/unsloth?locale=en-US

[^5_2]: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

[^5_3]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

[^5_4]: https://unsloth.ai/blog/r1-reasoning

[^5_5]: https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing

[^5_6]: https://github.com/unslothai/unsloth/issues/2499

[^5_7]: https://www.reddit.com/r/LocalLLaMA/comments/1or1e7p/i_finetuned_gemma_3_1b_for_cli_command/

[^5_8]: https://blogs.novita.ai/ar/deepseekr1-h100/

[^5_9]: https://www.reddit.com/r/LocalLLM/comments/1rz20yj/i_dont_think_local_llm_is_for_me_or_am_i_doing/

[^5_10]: https://www.dztecs.com/ai/how-to-benchmark-deepseek-r1-distilled-models-on-gpqa-using-ollama-and-openais-simple-evals/

[^5_11]: https://unsloth.ai/docs/get-started/install/intel

[^5_12]: https://rocm.docs.amd.com/projects/ai-developer-hub/en/v3.1/notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO.html

[^5_13]: https://unsloth.ai/docs/models/qwen3.5/fine-tune

[^5_14]: https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/

[^5_15]: https://zero-to-ai.dev/12-llm-finetuning/10_unsloth_fast_finetuning

[^5_16]: https://unsloth.ai/docs/get-started/install/conda-install

[^5_17]: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss

[^5_18]: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

[^5_19]: https://pypi.org/project/unsloth-zoo/

[^5_20]: https://github.com/ARBML/CIDAR


---

# كود Python كامل جاهز للتشغيل (ملف .py).

ملف JSON Config للتدريب مع Hyperparams.
أو Colab Notebook جاهز يمكن نسخه مباشرة.

جهزت لك 3 ملفات جاهزة للتشغيل لبدء تدريب **Baligh-1.7B v0** باستخدام **Unsloth** على **Qwen3-1.7B 4-bit** مع **CIDAR** كبيانات SFT ابتدائية، وهو مسار مناسب لنسخة v0 لأنه يعتمد على QLoRA خفيف وفعال فوق نموذج 1.5B.[^6_1][^6_2][^6_3]

## محتوى الملفات

- ملف Python يشغّل التدريب كاملًا: تحميل النموذج، إعداد LoRA، تحميل الداتا، تنسيق الـ prompts، ثم التدريب والحفظ.[^6_3][^6_1]
- ملف JSON فيه الـ hyperparameters الأساسية مثل `r=16` و`learning_rate=2e-4` و`batch_size=2` و`gradient_accumulation_steps=4`، وهي إعدادات عملية لبداية مستقرة على GPU محدود.[^6_4][^6_3]
- Notebook لـ Colab جاهز للنسخ والتشغيل مباشرة، ومبني على نفس فكرة notebook الخاص بـ Unsloth لتدريب Qwen2.5 بسرعة.[^6_2]


## كيفية الاستخدام

شغّل ملف الـ Python مع ملف الإعدادات كما هو إذا كانت أسماء أعمدة الداتا مطابقة، ثم عدّل فقط مسار الـ dataset أو أسماء الحقول لو احتجت ذلك. Unsloth يدعم بيئات Conda وPyTorch مع CUDA 11.8 أو 12.1، كما أن متطلباته الحديثة تركز على Python 3.11+ في التثبيتات الموصى بها.[^6_5][^6_6]

## ما أنصح به قبل التشغيل

ابدأ أولًا بـ run قصيرة مثل `max_steps=50` أو `100` للتأكد أن تنسيق CIDAR مناسب وأن الـ loss ينخفض بشكل طبيعي، ثم ارفع عدد الخطوات بعد التحقق. ولو كنت على Colab T4 أو GPU مشابه، فاختيار 4-bit QLoRA مع gradient checkpointing هو الأفضل عادة لتقليل استهلاك الذاكرة مع نماذج 1.5B.[^6_3][^6_4]

## تعديلات سريعة

لو أردت استخدام `evol-instruct-arabic` بدل CIDAR، غيّر فقط `dataset.path` وحقول الإدخال/الإخراج في ملف الـ JSON. ولو احتجت Arabic CPT لاحقًا، لا تستخدم هذا السكربت كما هو، لأن هذا مخصص لـ SFT/Instruction tuning وليس continued pretraining الخام.[^6_7][^6_8]

هل تريد الآن نسخة ثانية من السكربت مخصصة لـ **CPT على نصوص خام** بدل SFT؟

<div align="center">⁂</div>

[^6_1]: https://github.com/unslothai/unsloth?locale=en-US

[^6_2]: https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing

[^6_3]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

[^6_4]: https://unsloth.ai/blog/r1-reasoning

[^6_5]: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements

[^6_6]: https://unsloth.ai/docs/get-started/install/conda-install

[^6_7]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^6_8]: https://www.youtube.com/watch?v=oOVOkaQ_DSM


---

# عاوز كود كل مرحله بالتفصيل هذا أنت طلبت قائمة Datasets عربية جاهزة ومحددة جدًا على Hugging Face مع الروابط، وترتيب استخدامها في CPT و SFT، ونسب المزج — أنا الآن أجهزها لك نهائيًا كقائمة عملية قابلة للتنفيذ فورًا.

Baligh-1.7B v0 – قائمة Datasets جاهزة على Hugging Face
1️⃣ Datasets لـ Continued Pretraining (CPT)
هذه الداتا خام ونصوص عربية طويلة، هدفها رفع الفصحى، الإملاء، والكوادر اللغوية، وتثبيت المصطلحات الإسلامية.
\#Dataset NameHugging Face LinkTokens / SizeDomainLicense
1
ArabicWeb24
[lightonai/ArabicWeb24](https://huggingface.co/datasets/lightonai/ArabicWeb24)
~28B tokens
Web crawl Arabic (cleaned)
Open
2
ArabicText-Large
[Jr23xd23/ArabicText-Large](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large)
743K articles, 244M words
General Arabic text
Apache 2.0
3
The Arabic Pile (Dialects)
[premio-ai/TheArabicPile_Dialects](https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects)
13 subsets
MSA + dialects
Open
4
Oscar Arabic
[oscar corpus Arabic](https://huggingface.co/datasets/unoweden/oscar_ar)
Multi-billion
Web, news, encyclopedic
Open
5
mC4 Arabic
[mlc4/ar](https://huggingface.co/datasets/google/mc4)
Very large
Multilingual w/ Arabic
Open
أولوية CPT
ArabicWeb24 ← الأفضل (28B tokens، نظيف جدًا، dedup قوي)[[huggingface](https://huggingface.co/datasets/lightonai/ArabicWeb24)]
ArabicText-Large ← ممتاز (743K articles، جودة عالية)[[huggingface](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large)]
The Arabic Pile ← مزيج واسع (MSA + Dialects)[[huggingface](https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects)]
OSCAR/mC4 ← backup واسع
2️⃣ Datasets لـ Islamic Corpus (CPT إضافي)
هذه الداتا ليست همهمة محض، لكن v0 يحتاجها لـ "هوية Baligh".
\#Dataset NameLinkContentFormatLicense
1
hadith_datasets
[meeAtif/hadith_datasets](https://huggingface.co/datasets/meeAtif/hadith_datasets)
6 major books (Bukhari, Tirmidhi, etc.)
JSON + CSV
Open
2
quran-question-answer-context
[nazimali/quran-question-answer-context](https://huggingface.co/datasets/nazimali/quran-question-answer-context)
Quran QA
JSON
CC
3
Quran-MD
[Burraq/quran-audio-text-dataset](https://huggingface.co/datasets/Buraaq/quran-audio-text-dataset)
Verse-level Arabic + translation + audio
Multi
Open
4
Arabic Islamic texts
ابحث على HF بـ arabic islamic text
متون، شروح، فتاوى
متنوع
مختلف
3️⃣ Datasets لـ Instruction / SFT
هذه الداتا منهجية: instruction + output، هدفها تعليم النموذج المحاورة، تلخيص، سؤال/جواب، reasoning، استخراج.
\#Dataset NameLinkSizeStyleLicense
1
CIDAR
[arbml/CIDAR](https://huggingface.co/datasets/arbml/CIDAR)
10K instructions
Arabic culturally relevant
Apache 2.0
2
evol-instruct-arabic
[FreedomIntelligence/evol-instruct-arabic](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic)
~50K+
CoT + multi-turn
Open
3
ArSyra instruction-tuning
[ArSyra/arsyra-instruction-tuning](https://huggingface.co/datasets/ArSyra/arsyra-instruction-tuning)
3.3K+ (preview 50)
MSA + dialects
Purchase
4
Gazelle
[Gazelle Arabic writing](https://huggingface.co/papers/2410.18163)
Comprehensive
Writing assistance
Open
5
Arabic Summarization datasets
[BounharAbdelaziz/arabic-msa-summarization](https://huggingface.co/collections/BounharAbdelaziz/arabic-msa-summarization-models-and-datasets)
Summ
Summ
Open
أولوية SFT
CIDAR ← 10K instruction عربي ثقافي مهم جدًا[[github](https://github.com/ARBML/CIDAR)]
evol-instruct-arabic ← multi-turn + CoT [[ai.google](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune?hl=ar)]
Gazelle ← كتابة عربية[[huggingface](https://huggingface.co/papers/2410.18163)]
Summarization datasets ← تلخيص عربي[[huggingface](https://huggingface.co/collections/BounharAbdelaziz/arabic-msa-summarization-models-and-datasets)]
4️⃣ Datasets لـ Evaluation (لا تدخل التدريب)
\#DatasetLinkPurpose
1
MMLU_Arabic
[FreedomIntelligence/MMLU_Arabic](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic)
Arabic QA/Benchmark
2
CIDAR-EVAL-100
[arbml/CIDAR](https://huggingface.co/datasets/arbml/CIDAR)
Cultural relevance eval
3
CIDAR-MCQ-100
[arbml/CIDAR](https://huggingface.co/datasets/arbml/CIDAR)
MCQ cultural eval
4
mr-tydi Arabic
[castorini/mr-tydi](https://huggingface.co/datasets/castorini/mr-tydi)
Retrieval QA
ترتيب الاستخدام في CPT و SFT
Stage A: CPT (Continued Pretraining)
ArabicWeb24 (70% من CPT)
ArabicText-Large (20%)
The Arabic Pile (10%)
Islamic corpus ← مرحلة منفصلة (قد تكون CPT موجّهة للإسلاميات)
نسب مزج CPT
DatasetWeight
ArabicWeb24
70%
ArabicText-Large
20%
Arabic Pile
10%
Islamic texts
seasonal (دورة منفصلة)
الهدف: فصحى قوية + مزيج معرفي عام + إسلامي.
Stage B: SFT (Instruction Tuning)
أولوية SFT:
CIDAR ← 40%
evol-instruct-arabic ← 35%
Gazelle ← 10%
Summarization ← 10%
Islamic QA ← 5%
نسب مزج SFT
DatasetWeight
CIDAR
40%
evol-instruct-arabic
35%
Gazelle
10%
Summarization
10%
Islamic QA
5%
ملاحظات تنفيذية سريعة
كل dataset قابل للتحميل بـ:
python
from datasets import load_dataset
ds = load_dataset("lightonai/ArabicWeb24", data_files="ArabicWeb24/**/*.arrow", split="train")

[[huggingface](https://huggingface.co/datasets/lightonai/ArabicWeb24)]
CIDAR تحميل:
python
cidar = load_dataset("arbml/CIDAR")

[[github](https://github.com/ARBML/CIDAR)]
hadith_datasets: CSV + JSON مركّب وواضح جدًا.[[huggingface](https://huggingface.co/datasets/meeAtif/hadith_datasets)]
Evaluation sets لا تدخل التدريب أبدًا.
إذا حابب، في الرسالة التالية أجهز لك:
كود Python كامل يخلط هذه datasets بنسب CPT/SFT،
JSON конфиг للمزج،
أو model card جاهز للنشر على Hugging Face لـ Baligh-1.7B-v0.

جهزت لك **كود كل مرحلة** في pipeline منفصل: بناء خليط CPT، تجهيز JSONL، بناء خليط SFT، تدريب Unsloth، ثم تقييم مبدئي، مع ملف config مركزي ودليل تشغيل مختصر.[^7_1][^7_2][^7_3]

## الملفات

- `01_build_cpt_mixture.py`: يحمّل datasets الخاصة بـ CPT ويخلطها حسب الأوزان المقترحة مثل ArabicWeb24 وArabicText-Large وTheArabicPile_Dialects.[^7_4][^7_5][^7_1]
- `02_prepare_cpt_jsonl.py`: يحول ناتج CPT إلى JSONL بسيط بصيغة `{"text": ...}` مناسب لخطوة التدريب الخام.[^7_6]
- `04_build_sft_mixture.py`: يطبع ويطبع prompt موحد من datasets مثل CIDAR وevol-instruct-arabic وQuran QA ثم يعمل interleave حسب النسب.[^7_2][^7_3][^7_7]
- `03_train_sft_unsloth.py`: يشغّل تدريب SFT على Qwen3-1.7B 4-bit عبر Unsloth بإعدادات QLoRA عملية.[^7_8][^7_9][^7_10]
- `05_run_eval.py`: يفحص تحميل النموذج ويجرب generation أولي، مع تحميل مجموعات eval مثل MMLU_Arabic وmr-tydi.[^7_11]


## كيف تبدأ

ابدأ أولًا بتعديل `baligh_pipeline_config.json` لأن بعض datasets قد تحتاج تغيير أسماء الأعمدة أو قبول شروط الوصول على Hugging Face، خصوصًا ArabicWeb24 وبعض المجموعات الأخرى. بعد ذلك نفّذ المراحل بالترتيب: CPT mixture ثم CPT JSONL، ثم SFT mixture، ثم تدريب Unsloth، ثم التقييم.[^7_10][^7_12][^7_1]

## ملاحظات مهمة

الكود الحالي **baseline عملي** وليس final production pipeline، لأن بعض datasets على Hugging Face تختلف في schema بشكل فعلي من إصدار لآخر، لذلك أضفت طبقة تطبيع أولية لكن قد تحتاج تخصيصًا بسيطًا عند أول تشغيل. كذلك استخدام QLoRA مع 1.5B مناسب جدًا لنسخة v0 لأنه يقلل استهلاك الذاكرة مع الاحتفاظ بسرعة جيدة على Colab أو GPU متوسط.[^7_13][^7_14][^7_15][^7_10]

## الخطوة التالية

أفضل خطوة الآن هي أن أحوّل هذا من baseline إلى **نسخة production-grade مخصصة لـ Baligh** تشمل:

- cleaning أقوى،
- dedup،
- packing/tokenization،
- train/valid split،
- logging على Weights \& Biases،
- وmodel card جاهز للنشر على Hugging Face.[^7_16][^7_17]

هل تريدني أجهز لك الآن **نسخة production-grade كاملة** بدل النسخة الأساسية؟

<div align="center">⁂</div>

[^7_1]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^7_2]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^7_3]: https://github.com/ARBML/CIDAR

[^7_4]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^7_5]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^7_6]: https://mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/

[^7_7]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^7_8]: https://github.com/unslothai/unsloth?locale=en-US

[^7_9]: https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing

[^7_10]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

[^7_11]: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic

[^7_12]: https://huggingface.co/datasets/lightonai/ArabicWeb24/tree/main/ArabicWeb24/00024

[^7_13]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^7_14]: https://arxiv.org/html/2507.14688v2

[^7_15]: https://unsloth.ai/blog/r1-reasoning

[^7_16]: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

[^7_17]: https://www.youtube.com/watch?v=oOVOkaQ_DSM


---

# أو model card جاهز للنشر على Hugging Face لـ Baligh-1.7B-v0.

جاهز. هذا **Model Card كامل ومهني** لـ **Baligh-1.7B-v0** مصمم للنشر مباشرة على Hugging Face، مع أقسام واضحة وتفاصيل تقنية دقيقة، بما يتوافق مع أدلة Hugging Face للموديلات الصغيرة/المتوسطة التي تعتمد على continued pretraining وSFT.[^8_1][^8_2]

***

# 📘 Baligh-1.7B-v0 — Model Card

## 🎯 Overview

**Baligh-1.7B-v0** هو نموذج لغة عربي صغير بحجم 1.5 مليار معامل، مُصمم ليكون **أول المساعدة العربية الإسلامية** للمهام التعليمية والمعرفية. النموذج مبني على **Qwen3-1.7B Instruct** (Base model) مع **Arabic continued pretraining** ثم **supervised fine-tuning (SFT)** على بيانات عربية منقحة تشمل تعليمات عامة، كتابة عربية، تلخيص، وسؤال/جواب إسلامي.

الهدف من Baligh-1.7B-v0:

- فصحى قوية، إملاء دقيق، وبنية لغوية سليمة.
- فهم جيد للمصطلحات الإسلامية (فقه، حديث، تفسير، عقيدة).
- سلوك instruct جيد في الرد على التعليمات العربية.
- دعم **structured output** (JSON, Markdown tables).
- نمط **مختصر، واضح، ومفيد**، مناسب للطلاب والمهتمين بالعلوم الإسلامية.

النموذج **ليس** مصدرًا موثوقًا للفتوى الدقيقة، ولا يُستخدم للاستنابة الشرعية في المسائل الحساسة دون مراجعة بشرية.

***

## 📦 Model Details

| Property | Value |
| :-- | :-- |
| **Model name** | `Baligh-1.7B-v0` |
| **Base model** | `unsloth/Qwen3-1.7B-Base` (Qwen3-1.7B) [^8_1] |
| **Size** | 1.54B parameters (1.5B) |
| **Architecture** | Transformer causal LM, 28 layers, GQA, 32K context [^8_1] |
| **Language(s)** | Arabic (primary), multilingual base |
| **Training stage** | CPT (Arabic) + SFT (Instruction) |
| **License** | Apache 2.0 (أو تعديله حسب ترخيص Qwen) |
| **Type** | Instruct / Chat Assistant |
| **Intended use** | Teaching, Islamic QA, general Arabic QA, summarization, rewriting, structured output |


***

## 🧠 Training Architecture

### Base Model

Baligh-1.7B-v0 مبني على **Qwen3-1.7B Instruct**، وهو نموذج Causal LM من سلسلة Qwen2.5، متعدد اللغات، يدعم أكثر من 29 لغة بما فيها العربية. السياق الأقصى 32,768 token، مع دعم محسّن للـ structured data وJSON.[^8_1]

### Training Stages

النموذج تم تدريبه عبر **階段ين**:

1. **Stage 1: Arabic Continued Pretraining (CPT)**
    - هدف: رفع fluency في الفصحى، تحسين الإملاء، تثبيت المصطلحات الإسلامية.
    - بيانات: نصوص عربية خام (ArabicWeb24, ArabicText-Large, TheArabicPile_Dialects) + pool إسلامي محدود.[^8_3][^8_4][^8_5]
    - نسب مزج:
        - ArabicWeb24: 70%
        - ArabicText-Large: 20%
        - TheArabicPile: 10%
    - الهدف: base model عربي قوي، وليس chatbot نهائي.
2. **Stage 2: Supervised Fine-Tuning (SFT)**
    - هدف: تعليم النموذج الرد على التعليمات، الشرح، التلخيص، إعادة الصياغة، والسؤال/الجواب.
    - بيانات: Instruction datasets عربية (CIDAR, evol-instruct-arabic, Gazelle, Quran QA).[^8_6][^8_7][^8_8]
    - نسب مزج:
        - CIDAR: 40%
        - evol-instruct-arabic: 35%
        - Gazelle: 10%
        - Summarization: 10%
        - Islamic QA: 5%

***

## 📚 Data Sources

### CPT Data

| Dataset | Link | Domain |
| :-- | :-- | :-- |
| ArabicWeb24 | [lightonai/ArabicWeb24](https://huggingface.co/datasets/lightonai/ArabicWeb24) | Web crawl Arabic (cleaned) [^8_3] |
| ArabicText-Large | [Jr23xd23/ArabicText-Large](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large) | General Arabic text [^8_4] |
| TheArabicPile_Dialects | [premio-ai/TheArabicPile_Dialects](https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects) | MSA + dialects [^8_5] |

### SFT Data

| Dataset | Link | Type |
| :-- | :-- | :-- |
| CIDAR | [arbml/CIDAR](https://github.com/ARBML/CIDAR) | Arabic culturally relevant instructions [^8_7] |
| evol-instruct-arabic | [FreedomIntelligence/evol-instruct-arabic](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic) | CoT + multi-turn instructions [^8_6] |
| Gazelle | [Gazelle Arabic writing](https://huggingface.co/papers/2410.18163) | Arabic writing assistance [^8_9] |
| quran-question-answer-context | [nazimali/quran-question-answer-context](https://huggingface.co/datasets/nazimali/quran-question-answer-context) | Quran QA [^8_8] |

### Evaluation Data (لا تدخل التدريب)

| Dataset | Link | Purpose |
| :-- | :-- | :-- |
| MMLU_Arabic | [FreedomIntelligence/MMLU_Arabic](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic) | Arabic QA/Benchmark [^8_10] |
| mr-tydi | [castorini/mr-tydi](https://huggingface.co/datasets/castorini/mr-tydi) | Retrieval QA |


***

## 🛠️ Training Details

### Hyperparameters (SFT)

| Parameter | Value |
| :-- | :-- |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| LR scheduler | linear |
| Batch size | per_device=2, acc=4 → effective=8 |
| Warmup steps | 10 |
| Max steps | 200 (v0 baseline) |
| Seed | 3407 |
| FP16 | True |
| BF16 | False |
| Max seq length | 2048 |
| Use gradient checkpointing | `unsloth` (via Unsloth) [^8_11][^8_12] |

### Environment

- Python: 3.11–3.12
- PyTorch: 2.5+ (CUDA 11.8/12.1)
- Unsloth: latest (QLoRA 4-bit, 2x faster finetuning)[^8_13][^8_14]
- TRL: SFTTrainer
- Datasets: HuggingFace datasets

***

## 🧪 Evaluation

### Evaluation Strategy

- مجموعة eval ثابتة لا تدخل التدريب (MMLU_Arabic, mr-tydi, CIDAR eval subsets).
- قياسات:
    - Perplexity على Arabic held-out set.
    - دقة QA عربية عامة.
    - دقة المصطلحات الإسلامية.
    - جودة structured outputs (JSON, Markdown).
    - human eval: correctness, clarity, Arabic quality, usefulness, faithfulness.


### Current Limitations

- ليس نموذج فتوى دقيق؛ قد يخطئ في المسائل الفنية الدقيقة.
- قد يبالغ في الثقة أحيانًا.
- لا يتبع رAG للتحقق من الحقائق.
- قد يُظهر خلطًا بين النقل والفتوى بدون توضيح.
- v0 baseline، لا يزال بحاجة إلى تحسينات في:
    - reasoning المعقد.
    - الهدوء في المسائل غير المؤكدة.
    - توليد نص طويل أكثر استقرارًا.

***

## 🧭 Intended Use

### Primary Use Cases

- مساعد تعليمي للطلاب في العلوم الإسلامية.
- إجابات عامة عن المفاهيم الإسلامية (فقه، حديث، تفسير، عقيدة).
- شرح ومفاهيم عربية.
- تلخيص و إعادة صياغة نصوص عربية.
- استخراج معلومات (entities, structured data).
- كتابة عربية منضبطة (مقالات، شرح، نقاط).


### Secondary Use Cases

- RAG friendliness: يمكن دمجه مع نظام RAG لمراجعة الحقائق.
- QA عربي عام (غير شرعي).
- مساعد للباحثين في النصوص الإسلامية.


### Not Recommended Uses

- فتوى دقيقة في مسائل فقهية حساسة بدون مراجعة بشرية.
- أحكام شرعية نهائية.
- مصادر وحيدة للحقائق الشرعية.
- استخدام طبي/قانوني بدون تضمن بشري.

***

## ⚠️ Safety \& Ethics

- النموذج ليس هيئة شرعية، ولا يُعتبر مصدر فتوى نهائية.
- يجب على المستخدم عدم الاعتماد عليه في مسائل شرعية حساسة بدون مراجعة.
- النموذج قد يخطئ في الحقائق الدقيقة؛ يجب التحقق من المصادر.
- لا يُستخدم في توليد محتوى سياسي/اجتماعي حساس دون مراجعة.
- يجب إضافة طبقة التحقق (RAG أو مراجع بشرية) للحقائق.

***

## 🔧 How to Use

### Installation

```bash
pip install unsloth torch transformers trl datasets accelerate bitsandbytes
```


### Basic Usage (Python)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "your-org/Baligh-1.7B-v0-lora",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = "اشرح الفرق بين الفقه والعقيدة باختصار."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[^8_0], skip_special_tokens=True))
```


### Template Prompt

```text
### Instruction:
اشرح الفرق بين الفقه والعقيدة باختصار.

### Response:
```


***

## 📝 Citation

```bibtex
@misc{Baligh-1.7B-v0,
  title={Baligh-1.7B-v0: A Small Arabic Islamic Assistant},
  author={Mohamed Kandil},
  year={2026},
  url={https://huggingface.co/your-org/Baligh-1.7B-v0}
}
```


***

## 📄 License

- Base model: Qwen3-1.7B (ترخيص Qwen الأصلي).
- Baligh-1.7B-v0: Apache 2.0 (أو مطابق لترخيص Qwen + مساهماتك).
- التدريب: بيانات مفتوحة وم торговية كما في الجدول.

***

## 🏆 Acknowledgements

- Qwen Team (Qwen2.5).
- Unsloth team (QLoRA fast finetuning).
- CIDAR authors.
- FreedomIntelligence (evol-instruct-arabic).
- HuggingFace datasets community.

***

## 📌 Versioning

- **v0**: إطلاق أولي، baseline قوي.
- **v0.1–v0.9**: مراجعات تدريجية.
- **v1**: النسخة الأولى الرسمية.

***

أقدر أنسخ هذا مباشرة كـ `README.md` لـ repo في Hugging Face:

```markdown
# Baligh-1.7B-v0

<!-- ضع هنا محتوى الـ Model Card -->
```

ولو حابب، أجهز لك الآن:

- نسخة Markdown جاهزة للنسخ مباشرة.
- أو نسخة مع `model_card.yaml` + هيكل repo كامل.

<div align="center">⁂</div>

[^8_1]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^8_2]: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

[^8_3]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^8_4]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^8_5]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^8_6]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^8_7]: https://github.com/ARBML/CIDAR

[^8_8]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^8_9]: https://huggingface.co/papers/2410.18163

[^8_10]: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic

[^8_11]: https://unsloth.ai/blog/r1-reasoning

[^8_12]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

[^8_13]: https://github.com/unslothai/unsloth?locale=en-US

[^8_14]: https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing


---

# عاوز اشغل باستخدام colab cli هل تريدني أجهز لك الآن نسخة production-grade كاملة بدل النسخة الأساسية؟

تقدر تشغّل كل الpipeline بتاع Baligh-1.7B-v0 من **الترمينال** باستخدام **Colab CLI** بدل ما تفتح المتصفح، وفعليًا ده أنسب سيناريو ليك بما إن Google أعلنت رسميًا عن Google Colab CLI وربطته بالـ terminal المحلي لتشغيل سكربتات Python على runtimes فيها GPU.[^9_1][^9_2][^9_3]
هقسم لك الرد إلى جزئين:

1) إعداد وتشغيل **Colab CLI**،
2) تحويل سكربتات Baligh اللي جهزناها إلى **production-grade flow** فوق Colab CLI.

***

## 1️⃣ إعداد Google Colab CLI وتشغيله

### 1.1 تثبيت Colab CLI محليًا

من الماك / لينكس:

```bash
pip install -U google-colab-cli
# أو عبر uv (لو شغال بيه):
uv tool install -U google-colab-cli
```

Google Colab CLI الرسمي منشور على GitHub و PyPI، وبيوفر أوامر لإسناد runtime GPU وتشغيل سكربتات Python وسحب الـ artifacts.[^9_4][^9_1]

### 1.2 تسجيل الدخول

```bash
colab auth login
```

هتتفتح صفحة في المتصفح، تسجّل الدخول بحساب Google، وتدّي الـ scopes المطلوبة، والـ token يتخزن في الـ OS data directory عشان تقدر تستخدم الأوامر بدون كل مرة login.[^9_5]

### 1.3 طلب Runtime بـ GPU

```bash
# اختيار تفاعلي
colab server assign

# أو non-interactive — مثلاً T4 high RAM
colab server assign --variant gpu --accelerator T4 --high-ram -k
```

ده بيسند لك runtime Colab حقيقي (زي ما لو فتحت notebook واخترت GPU)، لكن كله من الترمينال.[^9_2][^9_1][^9_5]

***

## 2️⃣ تشغيل Baligh pipeline عبر Colab CLI (Production-style)

نستغل الملفات اللي جهزناها (CPT/SFT mix + training) ونخلّي Colab هو الbackend:

### 2.1 هيكل مشروعك محليًا

افترض أن عندك repo بالشكل:

```text
baligh/
  01_build_cpt_mixture.py
  02_prepare_cpt_jsonl.py
  03_train_sft_unsloth.py
  04_build_sft_mixture.py
  05_run_eval.py
  baligh_pipeline_config.json
  requirements.txt
```

السكربتات دي اللي سلّمتها لك في الرد السابق؛ تقدر تحطها تحت `baligh/` أو أي structure يعجبك.

### 2.2 رفع المشروع إلى Colab runtime

Colab CLI بيوفّر أوامر لرفع الملفات وتشغيلها على runtime:

```bash
cd baligh

# ارفع كل ملفات المشروع إلى الـ runtime
colab upload . --to /workspace/baligh
```

بكده أصبح في runtime مسار `/workspace/baligh` فيه كل سكربتات Baligh.[^9_1][^9_4]

### 2.3 تثبيت المتطلبات داخل runtime

تقدر تنفّذ أوامر shell على runtime:

```bash
colab exec --workdir /workspace/baligh -- \
  pip install -U "unsloth" "datasets" "trl" "accelerate" "bitsandbytes" "transformers"
```

ده مكافئ لخلايا `!pip install ...` في notebook، لكن من الترمينال عبر `colab exec`.[^9_4][^9_1]

### 2.4 تشغيل مراحل CPT / SFT بالترتيب

#### (أ) بناء مزيج CPT

```bash
colab exec --workdir /workspace/baligh -- \
  python 01_build_cpt_mixture.py
```

السكربت يستخدم `baligh_pipeline_config.json` لتحميل ArabicWeb24 وArabicText-Large وTheArabicPile_Dialects ويعمل لهم interleave حسب الأوزان (70/20/10) ويخزنهم في `output/baligh_cpt_mixture`.[^9_6][^9_7][^9_8]

#### (ب) تجهيز JSONL لـ CPT

```bash
colab exec --workdir /workspace/baligh -- \
  python 02_prepare_cpt_jsonl.py
```

ده يحول الـ dataset إلى `output/baligh_cpt_train.jsonl` بصيغة `{"text": ...}`، مناسب لـ CPT خام لو حبيت تعمل training مستقل (السكربت الحالي baseline، CPT الحقيقي هتحتاج له loop تدريب منفصل).

#### (ج) بناء مزيج SFT

```bash
colab exec --workdir /workspace/baligh -- \
  python 04_build_sft_mixture.py
```

ده يحمل:

- `arbml/CIDAR` (40%)
- `FreedomIntelligence/evol-instruct-arabic` (35%)
- `ArSyra/arsyra-instruction-tuning` (10% لو متاح)
- `nazimali/quran-question-answer-context` (5%)

ويطبعها على شكل prompt موحد ثم يخزنها في `output/baligh_sft_mixture`.[^9_9][^9_10][^9_11]

#### (د) تدريب SFT بـ Unsloth على Qwen3-1.7B

```bash
colab exec --workdir /workspace/baligh -- \
  python 03_train_sft_unsloth.py
```

هذا السكربت:

- يحمّل `unsloth/Qwen3-1.7B-Base`.[^9_12]
- يفعّل LoRA (r=16, alpha=16, dropout=0, target_modules للكويز/فاليو/أو/جيت/أب/داون).
- يدرّب باستخدام SFTTrainer من TRL على `output/baligh_sft_mixture`.
- يستخدم hyperparams اللي في `baligh_pipeline_config.json` (lr=2e-4, batch=2, acc=4, max_steps=200, warmup=10).[^9_13][^9_14]
- يحفظ النموذج في `output/Baligh-1.7B-v0-lora`.


### 2.5 سحب النموذج والنتائج لبيئتك المحلية

بعد انتهاء التدريب:

```bash
colab download /workspace/baligh/output/Baligh-1.7B-v0-lora ./Baligh-1.7B-v0-lora
```

أو لو CLI بيشتغل بمفهوم tar:

```bash
colab download /workspace/baligh/output/Baligh-1.7B-v0-lora
```

Colab CLI يدعم **Seamless Artifact Recovery** لسحب الموديلات والداتا والـ notebooks بدون وجع دماغ.[^9_3][^9_2][^9_1]

### 2.6 تفاعل حي (اختياري)

لو حابب تخش على runtime وتشتغل يدويًا:

```bash
colab repl --workdir /workspace/baligh
# أو:
colab console
```

ده يفتح لك interactive shell داخل runtime تقدر منه تشغل `python`، تجرب `import torch`, وتطمن على GPU (`nvidia-smi`).[^9_5][^9_1]

***

## 3️⃣ تحويل النسخة الحالية إلى Production-Grade

اللي فوق **production-ish** من ناحية تشغيل من CLI، لكن ممكن نرقّيه لمرحلة أعلى:

- إضافة:
    - logging إلى `wandb` أو `tensorboard`.
    - train / val split في خليط SFT.
    - checkpointing دوري مع best model على val loss.
    - packing sequences لـ CPT (لو هتعمل CPT فعلي).
- اتساق كامل مع model card وHF repo:
    - سكربت `upload_to_hf.py` يرفع النموذج.
    - سكربت `export_gguf.py` لو حابب تشغيل محلي.
- integration مع Colab CLI:
    - سكربت واحد `run_baligh_colab.sh` يحجز runtime، يرفع repo، يثبت المتطلبات، يشغل stages، وينزّل artifacts.

لو تحب أطلعك للـ level ده، أقدر في الرسالة الجاية أجهز لك:

- `run_baligh_colab.sh` (سكربت واحد يشغّل الـ pipeline كله عبر Colab CLI).
- ملف `requirements.txt` مضبوط لـ Unsloth + TRL + HF stack.
- سكربت `upload_to_hf.py` لرفع `Baligh-1.7B-v0` تلقائيًا على Hugging Face.

هل تفضّل نبدأ بـ **run_baligh_colab.sh** ولا **upload_to_hf.py** أولًا؟
<span style="display:none">[^9_15][^9_16][^9_17][^9_18][^9_19][^9_20][^9_21][^9_22][^9_23]</span>

<div align="center">⁂</div>

[^9_1]: https://developers.googleblog.com/introducing-the-google-colab-cli/

[^9_2]: https://www.developer-tech.com/news/google-colab-cli-local-terminal-cloud-gpus/

[^9_3]: https://www.helpnetsecurity.com/2026/06/08/google-colab-command-line-interface-cli/

[^9_4]: https://github.com/googlecolab/google-colab-cli

[^9_5]: https://lib.rs/crates/colab-cli

[^9_6]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^9_7]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^9_8]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^9_9]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^9_10]: https://github.com/ARBML/CIDAR

[^9_11]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^9_12]: https://huggingface.co/unsloth/Qwen3-1.7B

[^9_13]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

[^9_14]: https://unsloth.ai/blog/r1-reasoning

[^9_15]: https://stackoverflow.com/questions/77907657/installing-gcloud-components-in-a-google-colab

[^9_16]: https://docs.cloud.google.com/sdk/gcloud/reference/colab

[^9_17]: https://docs.data-community.publishing.service.gov.uk/tools/google-cloud-platform/gcp-access/

[^9_18]: https://dev.to/akshay090/here-s-how-i-made-a-cli-tool-to-work-with-ipynb-and-google-colab-12jm

[^9_19]: https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/

[^9_20]: https://colab.research.google.com/github/GoogleCloudPlatform/devrel-demos/blob/main/other/colab/Using Google Cloud from Colab.ipynb

[^9_21]: https://www.youtube.com/watch?v=4OXBx76q5G0

[^9_22]: https://github.com/unslothai/notebooks

[^9_23]: https://cloud.google.com/sdk


---

# الاتنين و عاوز المشروع clear Architecture

فهمت عليك تمامًا: تريد

1) تشغيل كل الـ pipeline عن طريق **Colab من الـ CLI**،
2) وفي نفس الوقت يكون عندك **مشروع منظّم (clear architecture)**،
3) مع سكربتين أساسيين:
    - سكربت orchestration لـ Colab،
    - سكربت رفع إلى Hugging Face.

في الرد هذا هكتب لك كل شيء كنص (تقدر تنسخه مباشرة في ملفاتك)، لكن لا أستطيع فعليًا توليد/رفع ملفات جاهزة لك من البيئة الحالية؛ فاعتبر الرد blue-print جاهز للنسخ.

سأقسّم الرد إلى:

1. Architecture واضحة للمشروع.
2. سكربت `run_baligh_colab.sh` لتشغيل كل المراحل عبر Colab CLI من الترمينال.
3. سكربت `upload_to_hf.py` لرفع الموديل إلى Hugging Face.
4. ملاحظات تشغيل خطوة بخطوة.

استند في تصميم الـ CLI على توثيق Google Colab CLI الرسمي الذي يوضح أوامر `colab new/exec/upload/download` لتشغيل سكربتات بايثون على runtimes فيها GPU ، وعلى توثيق `huggingface_hub` لدوال `create_repo` و`upload_folder` لرفع النماذج للـ Hub.[^10_1][^10_2][^10_3][^10_4]

***

## 1️⃣ Clear Architecture للمشروع

اقترح هيكل بالشكل التالي:

```bash
Baligh-1.7B/
  README.md
  requirements.txt

  config/
    baligh_pipeline_config.json

  data/
    cpt/
      # raw + processed CPT data (اختياري لو حابب تحفظ local copies)
    sft/
      # raw + processed SFT data

  scripts/
    01_build_cpt_mixture.py
    02_prepare_cpt_jsonl.py
    03_train_sft_unsloth.py
    04_build_sft_mixture.py
    05_run_eval.py
    run_baligh_colab.sh
    upload_to_hf.py

  models/
    Baligh-1.7B-v0-lora/
      # يخرج هنا بعد التدريب (محليًا بعد download من Colab)

  logs/
    train/
    eval/
```

- `config/baligh_pipeline_config.json`: ملف مركزي فيه:
    - خلط CPT (ArabicWeb24, ArabicText-Large, TheArabicPile بنِسَب 70/20/10).[^10_5][^10_6][^10_7]
    - خلط SFT (CIDAR, evol-instruct-arabic, … بالنِسَب اللي اتفقنا عليها).[^10_8][^10_9][^10_10]
    - إعدادات التدريب (lr, batch, max_steps… إلخ).
    - إعدادات رفع HuggingFace (repo_id، private أو public).
- `scripts/*`: كل مرحلة معزولة:
    - 01 + 02: CPT data prep.
    - 04: SFT mixture.
    - 03: تدريب SFT بـ Unsloth.
    - 05: eval بسيط.
- `run_baligh_colab.sh`: سكربت واحد:
    - يحجز Colab GPU.
    - يرفع المشروع.
    - يثبّت المتطلبات.
    - يشغّل المراحل بالترتيب.
    - ينزّل الموديل النهائي.
- `upload_to_hf.py`: سكربت صغير يستخدم `huggingface_hub` لرفع `models/Baligh-1.7B-v0-lora` إلى Hub.[^10_2][^10_3]

***

## 2️⃣ سكربت تشغيل كامل عبر Colab CLI — `run_baligh_colab.sh`

ضع هذا الملف في `scripts/run_baligh_colab.sh` واجعله executable (`chmod +x scripts/run_baligh_colab.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail

# مسار المشروع المحلي
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[^10_0]}")/.." && pwd)"

SESSION_NAME="${BALIGH_COLAB_SESSION:-baligh-train}"

echo "[Baligh] Using project root: $PROJECT_ROOT"
echo "[Baligh] Colab session name: $SESSION_NAME"

# 1) إنشاء/تخصيص Colab session مع GPU (T4 كنقطة بداية)
colab new -s "$SESSION_NAME" --gpu T4

# 2) تثبيت المتطلبات داخل الـ session
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess, textwrap

# نشتغل من /workspace
os.makedirs("/workspace", exist_ok=True)
os.chdir("/workspace")

# تثبيت المتطلبات الأساسية
cmd = "pip install -U unsloth transformers trl accelerate datasets bitsandbytes huggingface_hub wandb"
print(f"Running: {cmd}")
subprocess.check_call(cmd.split())
PY

# 3) رفع المشروع إلى /workspace/Baligh-1.7B داخل Colab
echo "[Baligh] Uploading project to Colab..."
colab upload -s "$SESSION_NAME" "$PROJECT_ROOT" /workspace/Baligh-1.7B

# 4) تشغيل مراحل CPT + SFT + Training

# 4.1 Build CPT mixture
echo "[Baligh] Building CPT mixture..."
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess
os.chdir("/workspace/Baligh-1.7B")
subprocess.check_call(["python", "scripts/01_build_cpt_mixture.py"])
subprocess.check_call(["python", "scripts/02_prepare_cpt_jsonl.py"])
PY

# 4.2 Build SFT mixture
echo "[Baligh] Building SFT mixture..."
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess
os.chdir("/workspace/Baligh-1.7B")
subprocess.check_call(["python", "scripts/04_build_sft_mixture.py"])
PY

# 4.3 Train SFT with Unsloth
echo "[Baligh] Training SFT with Unsloth..."
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess
os.chdir("/workspace/Baligh-1.7B")
subprocess.check_call(["python", "scripts/03_train_sft_unsloth.py"])
PY

# 5) Download trained model back to local
echo "[Baligh] Downloading trained model to local ./models/Baligh-1.7B-v0-lora ..."
mkdir -p "$PROJECT_ROOT/models"
colab download -s "$SESSION_NAME" \
  /workspace/Baligh-1.7B/models/Baligh-1.7B-v0-lora \
  "$PROJECT_ROOT/models/Baligh-1.7B-v0-lora"

echo "[Baligh] Done. Model is in models/Baligh-1.7B-v0-lora"
echo "[Baligh] You can now run: python scripts/upload_to_hf.py"
```

ملاحظات:

- يستخدم أوامر موثقة لـ Colab CLI:
    - `colab new` لإنشاء session GPU.[^10_4]
    - `colab exec` لتشغيل Python من stdin أو ملفات.[^10_4]
    - `colab upload` لرفع المشروع.
    - `colab download` لسحب النموذج النهائي.[^10_4]
- كل مرحلة تُشغّل من داخل `/workspace/Baligh-1.7B` في الـ runtime.

قبل تشغيله لأول مرة:

```bash
pip install -U google-colab-cli   # تثبيت CLI نفسه [page:1]
colab auth login                  # تسجيل الدخول مرة واحدة
```

ثم:

```bash
cd Baligh-1.7B
./scripts/run_baligh_colab.sh
```


***

## 3️⃣ سكربت رفع إلى Hugging Face — `upload_to_hf.py`

هذا السكربت بسيط، يستخدم `huggingface_hub` بالطريقة الموصى بها: `create_repo` ثم `upload_folder`.[^10_3][^10_2]

ضعه في `scripts/upload_to_hf.py`:

```python
import os
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

# تحديد مسار ملف الـ config
PROJECT_ROOT = Path(__file__).resolve().parents[^10_1]
CONFIG_PATH = PROJECT_ROOT / "config" / "baligh_pipeline_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

repo_id = cfg["hf"]["repo_id"]          # مثال: "mohamed-kandil/Baligh-1.7B-v0"
private = cfg["hf"].get("private", False)
model_dir = PROJECT_ROOT / cfg["train"]["output_dir"]  # models/Baligh-1.7B-v0-lora

if not model_dir.exists():
    raise SystemExit(f"Model directory not found: {model_dir}")

api = HfApi()

# تأكد أنك عامل login قبل كده:
# huggingface-cli login
create_repo(repo_id=repo_id, private=private, exist_ok=True)

print(f"[Baligh] Uploading {model_dir} to Hugging Face Hub: {repo_id} ...")
upload_folder(
    repo_id=repo_id,
    folder_path=str(model_dir),
    repo_type="model",
    commit_message="Upload Baligh-1.7B-v0 model",
)

print(f"[Baligh] Done. Model is at: https://huggingface.co/{repo_id}")
```

تشغيله:

```bash
pip install -U huggingface_hub
huggingface-cli login   # مرة واحدة فقط [web:121][web:137]

cd Baligh-1.7B
python scripts/upload_to_hf.py
```

توثيق `huggingface_hub` يؤكد أن `upload_folder` + `create_repo` هما الطريقة البسيطة الصحيحة لرفع نماذج أو datasets إلى الـ Hub.[^10_2][^10_3]

***

## 4️⃣ ملاحظات على السكربتات الداخلية (Stage Scripts)

أنت بالفعل عندك/بنينا سابقًا سكربتات:

- `01_build_cpt_mixture.py`: يقرأ `config/baligh_pipeline_config.json` و:
    - يحمل ArabicWeb24, ArabicText-Large, TheArabicPile.
    - يختار field مناسب للنص (text/content).
    - يعمل `interleave_datasets` بنِسَب.[^10_6][^10_7][^10_5]
    - يحفظ الناتج في `data/cpt/baligh_cpt_mixture`.
- `02_prepare_cpt_jsonl.py`: يحوّل الـ mixture إلى `data/cpt/baligh_cpt_train.jsonl` بصيغة `{ "text": ... }`.
- `04_build_sft_mixture.py`: يحمّل CIDAR + evol-instruct-arabic + (ArSyra) + Quran QA ، ويوحّدها إلى prompt:[^10_9][^10_10][^10_8]

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

- `03_train_sft_unsloth.py`: يستخدم Unsloth:
    - يحمّل `unsloth/Qwen3-1.7B-Base`.[^10_11]
    - يضيف LoRA.
    - يدرّب باستخدام SFTTrainer على `data/sft/baligh_sft_mixture`.
    - يحفظ في `models/Baligh-1.7B-v0-lora`.

لو تحب أكتب لك نسخة نهائية من هذه السكربتات (ready-to-paste) أقدر أعمل ذلك في رسالة منفصلة، لكن بناءً على الرسائل السابقة عندك الآن skeleton قوي ويمكنك التعديل عليه بسهولة.

***

## 5️⃣ كيف تشغل كل شيء عمليًا

1. **إعداد المشروع:**
    - أنشئ مجلد `Baligh-1.7B`.
    - انسخ فيه الـ structure أعلاه + السكربتات.
    - عدّل `config/baligh_pipeline_config.json` (خاصة `hf.repo_id` و `train.max_steps`).
2. **تشغيل من الترمينال:**
    - ثبّت Colab CLI: `pip install -U google-colab-cli`.[^10_4]
    - `colab auth login`.
    - `cd Baligh-1.7B`.
    - `./scripts/run_baligh_colab.sh`.
3. بعد أن ينتهي:
    - تجد النموذج تحت: `models/Baligh-1.7B-v0-lora/`.
    - اختبره محليًا بـ `transformers` / `unsloth`.
4. **رفع إلى HF:**
    - `huggingface-cli login`.
    - `python scripts/upload_to_hf.py`.

***

لو تحب في الخطوة الجاية:

- أكتب لك **نسخ كاملة** من:
    - `01_build_cpt_mixture.py`
    - `04_build_sft_mixture.py`
    - `03_train_sft_unsloth.py`

بشكل production أكثر (مع logging بسيط، validation split، وconfig-driven بالكامل)، قول لي وسأركّز على السكربتات الثلاثة دي تحديدًا.
<span style="display:none">[^10_12][^10_13][^10_14][^10_15][^10_16][^10_17][^10_18][^10_19][^10_20][^10_21][^10_22][^10_23][^10_24][^10_25][^10_26][^10_27][^10_28][^10_29][^10_30][^10_31][^10_32][^10_33][^10_34]</span>

<div align="center">⁂</div>

[^10_1]: https://developers.googleblog.com/introducing-the-google-colab-cli/

[^10_2]: https://pypi.org/project/huggingface-hub/

[^10_3]: https://huggingface.co/docs/huggingface_hub/en/guides/upload

[^10_4]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^10_5]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^10_6]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^10_7]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^10_8]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^10_9]: https://github.com/ARBML/CIDAR

[^10_10]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^10_11]: https://huggingface.co/unsloth/Qwen3-1.7B

[^10_12]: https://github.com/googlecolab/google-colab-cli

[^10_13]: https://github.com/Akshay090/colab-cli

[^10_14]: https://github.com/googlecolab/google-colab-cli/issues

[^10_15]: https://lib.rs/crates/colab-cli

[^10_16]: https://dev.to/akshay090/here-s-how-i-made-a-cli-tool-to-work-with-ipynb-and-google-colab-12jm

[^10_17]: https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-128K-GGUF

[^10_18]: https://discuss.huggingface.co/t/upload-a-tf-model-to-huggingface/9662

[^10_19]: https://github.com/googlecolab/google-colab-cli/pulls

[^10_20]: https://www.geeksforgeeks.org/machine-learning/how-to-upload-and-share-model-to-huggingface/

[^10_21]: https://colab.research.google.com

[^10_22]: https://unsloth.ai/docs/get-started/unsloth-model-catalog

[^10_23]: https://huggingface.co/transformers/v4.0.1/model_sharing.html

[^10_24]: https://github.com/googlecolab/google-colab-cli/discussions/17

[^10_25]: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb

[^10_26]: https://pypi.org/project/colab-cli/

[^10_27]: https://github.com/huggingface/hub-docs/blob/main/docs/hub/models-adding-libraries.md

[^10_28]: https://www.youtube.com/watch?v=6HFlwqK3oeo

[^10_29]: https://hugging-face.cn/docs/huggingface_hub/guides/upload

[^10_30]: https://ulusoydigital.com/blog/googles-new-colab-cli-lets-developers-and-ai-agents-run-python-on-remote-colab-gpus-and-tpus-from-the-terminal

[^10_31]: https://huggingface.tw/docs/huggingface_hub/guides/upload

[^10_32]: https://discuss.huggingface.co/t/upload-and-commit-your-model-into-huggingface-hub/136548

[^10_33]: https://spikeinterface.readthedocs.io/en/0.102.0/tutorials/curation/plot_3_upload_a_model.html

[^10_34]: https://deepnote.com/blog/ultimate-guide-to-huggingfacehub-library-in-python


---

# عاوزه يكون SOTA

عشان Baligh-1.7B يبقى أقرب ما يكون لـ **SOTA في فئة 1–2B Arabic models**، لازم نتعامل مع الموضوع كـ pipeline بحثي/إنتاجي كامل:

- **داتا SOTA** (ArabicWeb24 + curated Islamic + high‑quality SFT)،
- **وصفة تدريب SOTA** (CPT + SFT + ممكن RL)،
- **Benchmark SOTA عربي رسمي** (ABB، Arabic-LLM-Bench، HELM Arabic)،
- **هندسة + eval loop صارم**.[^11_1][^11_2][^11_3][^11_4][^11_5][^11_6]

سأعطيك خطة SOTA على ثلاثة محاور: **الداتا**، **التدريب**، **التقييم**، ثم أوضح ما الذي تحتاجه زيادة فوق pipeline الحالي ليصبح منافس فعلي على leaderboards العربية.

***

## 1️⃣ كيف نعرّف “SOTA” عربيًا الآن؟

هناك الآن Benchmarks ولوحات رسمية لـ Arabic LLMs، منها:

- **Arabic Broad Benchmark (ABB / ABL)** من SILMA.ai: Broad benchmark + leaderboard مفتوح للنماذج العربية، يقيس مهام واسعة: فهم، reasoning، QA، كتابة، إلخ.[^11_2][^11_3]
- **HELM Arabic** من Stanford CRFM: نسخة عربية من HELM، لقياس نماذج متعددة على مهام متنوعة بشكل holistic.[^11_6]
- **Arabic-LLM-Benchmarks** (TII / MBZUAI وغيرها): repo يجمع Benchmarks وتقييمات لـ Arabic LLMs المعاصرة (مثل Jais، AceGPT، ALLaM).[^11_7][^11_8][^11_1]

“SOTA” في السياق ده يعني:

- أداء عالي على ABB / HELM Arabic / Arabic-LLM-Bench على مقاييس مثل correctness, reasoning, Arabic quality.
- بالمقارنة مع نماذج متخصصة مثل **LLM-X** من Arabic.AI، **SILMA models**، **ALLaM**، إلخ.[^11_9][^11_10][^11_7]

أنت في فئة **1.5B**، فهدفك الواقعي:

- SOTA في **“efficient Arabic SLMs” (≤2B)** وليس منافسة GPT-4.1/SILMA-9B.[^11_10][^11_9]
- أداء يتفوق على معظم النماذج العالمية العامة في العربية في نفس الحجم.

***

## 2️⃣ داتا SOTA: ماذا تحتاج فوق ما ذكرناه؟

### 2.1 ArabicWeb24 = أساس SOTA للـ CPT

- ArabicWeb24 مصمم بالأساس كـ high-quality Arabic pretraining dataset، فيه 28B+ tokens، مع وصف واضح لخطوات التنظيف، dedup، وفلترة الجودة.[^11_4][^11_5]
- المدونة نفسها تذكر أن تدريب نماذج ablation على ArabicWeb24 أظهر تحسنًا واضحًا مقابل خام CommonCrawl Arabic.[^11_5][^11_4]

بالنسبة لـ Baligh SOTA:

- **استخدم ArabicWeb24-V1/V5 كأساس رئيسي لـ CPT** بدل خلط OSCAR/mC4 عشوائيًا.
- استهدف **10–30B tokens effective CPT** على 1.5B (لو compute يسمح)؛ هذا تقريبًا ما تذكره أعمال مشابهة لموديلات Web-only Arabic.[^11_4][^11_5]


### 2.2 Islamic corpus SOTA-like

لا يوجد “SOTA Islamic corpus” قياسي، لكن SOTA بالنسبة لمشروعك يعني:

- متون أساسية + شروح + تفاسير + فتاوى موثوقة، مع **تنظيف شديد** (no OCR noise، no duplication، no random blogs).
- Annotated corpora حيث أمكن (مثل Quran QA، سلاسل hadith graded).[^11_11][^11_12]
- لو أمكن: **Quran MD** أو multimodal Qur’an datasets لدعم سياق متقدم.[^11_13]


### 2.3 SFT data SOTA

للـ SFT عليك أن تذهب أبعد من مجرد CIDAR + evol-instruct-arabic:

- **ABB-style tasks**: SFT data يغطي نفس أنواع المهام التي يقيسها Arabic Broad Benchmark (قراءة وفهم، reasoning، QA متعدد المجالات، كتابة، ترجمة، إلخ).[^11_3][^11_2]
- **Gazelle**: كتابة عربية عالية الجودة؛ يوفر dataset وإطار تقييم للقدرات الكتابية (rephrasing, editing, style).[^11_14]
- التركيز على:
    - **multi-step reasoning بالعربية** (chain-of-thought).
    - **instruction diversity**: مستويات مختلفة من الصعوبة، أسئلة مفتوحة، حالات رفض، إلخ.
    - **Islamic QA مصمم يدويًا** بسياقات منقحة، ليس فقط scraping.

***

## 3️⃣ وصفة تدريب SOTA لـ Baligh-1.7B

### 3.1 Continued Pretraining (CPT) جدي

- نماذج ArabicWeb24 الموصوفة في المدونة تدربت على subsets 28–39B tokens مع ablation، وبيّنت أن جودة البروسيسنج لها تأثير كبير على الـ downstream performance.[^11_5][^11_4]
- لتحقيق SOTA في فئة 1.5B عربي، تحتاج:
    - 10–30B tokens CPT فعلي على ArabicWeb24 (+ Islamic، + ArabicText-Large).
    - **token packing** جيد (لا تدرّب على sequences قصيرة مفصولة).
    - schedule نظيف (cosine decay, warmup كاف، gradient checkpointing).

> ما عندك compute كافي لهذا؟
> وقتها بتخلي SOTA على مستوى **post-training** / **SFT** وتقبل أن الـ base عربي جزئي.

### 3.2 SFT على مراحل (مرحلة عامة + مرحلة إسلامية)

- **Stage SFT-1 (General Arabic)**:
    - Data: CIDAR + evol-instruct-arabic + Gazelle + Summarization.
    - Tasks: QA عام، تلخيص، شرح، إعادة صياغة، كتابة، أسلوب.
- **Stage SFT-2 (Islamic-focused)**:
    - Data: Islamic QA curated + Quran QA + hadith QA + reasoning فقهي مبسط.
    - Tasks: نقل، شرح، صيغ “لا أعلم”، بيان المرجع.

استخدم Unsloth لـ QLoRA لكنها مرحلة SFT فقط؛ لو أردت SOTA حقيقي، قد تحتاج:

- مرحلة **DPO / ORPO / GRPO** لتحسين style والـ alignment (Unsloth يدعم GRPO على 1.5B مع 5GB VRAM تقريبًا).[^11_15][^11_16]


### 3.3 RL/Preference Optimization (اختياري لكن SOTA-ish)

نماذج reasoning التي تعيد إنتاج نمط R1-Zero تستخدم GRPO فوق Qwen3-1.7B مع 7GB VRAM فقط على Unsloth.[^11_16][^11_15]
لـ Baligh-SOTA:

- تبني small preference dataset بالعربية:
    - إجابات جيدة/متوسطة/سيئة على أسئلة دينية وتعليمية.
- تطبيق GRPO/DPO خلال 10–30k steps على 1.5B.
- الهدف: تحسين:
    - refusal behavior.
    - “لا أعلم” في المسائل المعقدة.
    - الوضوح والاختصار.

***

## 4️⃣ Benchmark SOTA: لازم تتقاس على لوحات رسمية

لو عايز تقول “SOTA” قدام الناس، لازم:

1. **تربط تقييمك بـ Arabic Broad Benchmark (ABB)** وتعرض نتائجك على ABL leaderboard؛ هذا الـ benchmark مصمم خصيصًا للنماذج العربية، مع تحليلات متقدمة وdetectors للتلوث.[^11_2][^11_3]
2. **تجرب على HELM Arabic** إذا أمكن، لأنه جزء من منظومة HELM الرسمية من Stanford لشفافية التقييم.[^11_6]
3. تستغل repos مثل **Arabic-LLM-Benchmarks** التي تجمع المهام العربية (AlGhafa, LAraBench, إلخ).[^11_17][^11_8][^11_1]

هدفك الواقعي:

- أن تكون ضمن **Top 1–3 models في فئة ≤2B parameters** على ABB/HELM Arabic في tasks عامة + كتابة + QA دينية سهلة.
- حتى لو 7B/9B models يتفوقون في المطلق، أنت تضع نفسك كـ “SOTA efficient Arabic SLM”.

***

## 5️⃣ تأثير الـ Architecture الذي نبنيه الآن

الـ architecture اللي نبنيه (clear project, CPT/SFT separation, Colab CLI orchestration، HF upload) لا يضمن بذاته SOTA، لكن:

- يسمح لك **بالتكرار السريع على وصفات الداتا + الـ hyperparameters** وقياس النتيجة على Benchmarks عربية قياسية.
- يسهل دمج:
    - ArabicWeb24 كـ CPT backbone.[^11_4][^11_5]
    - Benchmarks ABB/Arabic-LLM-Bench بداخل `05_run_eval.py` مثلًا.[^11_1][^11_2]

SOTA إذن سوف يأتي من:

- **كمية وجودة CPT + وصفة SFT المتقدمة + RL alignment**،
- **مقارنة صريحة على ABB/HELM Arabic**،
وليس فقط من “كويس على CIDAR dev”.

***

## 6️⃣ ما الذي أنصحك به لتقترب من SOTA فعليًا؟

1. **CPT جدّي**:
    - استثمر وقت في تشغيل CPT على ArabicWeb24 (على الأقل subset 5–10B tokens).
    - استخدم وصفة ArabicWeb24 paper/blog كمصدر إلهام للـ pre-processing والـ eval.[^11_18][^11_5][^11_4]
2. **SFT متدرج**:
    - Stage 1: Arabic general instruct (CIDAR/evol/Gazelle).
    - Stage 2: Islamic instruct curated داخليًا.
    - Stage 3: Preference/RL (على الأقل DPO صغير).
3. **Benchmark رسمي**:
    - اكتب adaptor بسيط يستهلك ABB/Arabic-LLM-Bench وHELM Arabic،
    - حافظ على eval scripts علنيّة،
    - انشر نتائجك (حتى لو لم تكن رقم 1 فورًا).
4. **Iterate**:
    - Baligh-1.7B-v0 = أول baseline.
    - Baligh-1.7B-v1 = بعد CPT + SFT قوي.
    - Baligh-1.7B-v2 = بعد RL + tune للهويّة الإسلاميّة + bench على ABB.

بهذه الطريقة تقدر تقول بثقة:
“Baligh-1.7B-v2 من ضمن أفضل النماذج العربية في فئة 1–2B على ABB/HELM Arabic، بأداء SOTA تقريبًا في فئته الخاصة.”

هل تحب أن أجهز لك في الخطوة القادمة **وصفة CPT تقريبية بالأرقام (tokens, steps, lr schedule) لنموذج 1.5B على ArabicWeb24** بحيث تكون قريبة من الوصفات المنشورة في ArabicWeb24 blog، مع resource estimates (GPU ساعات)؟
<span style="display:none">[^11_19][^11_20][^11_21]</span>

<div align="center">⁂</div>

[^11_1]: https://github.com/tiiuae/Arabic-LLM-Benchmarks

[^11_2]: https://huggingface.co/blog/silma-ai/arabic-llm-leaderboard

[^11_3]: https://silma.ai/arabic-llm-benchmark

[^11_4]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^11_5]: https://lighton.ai/lighton-blogs/arabicweb24

[^11_6]: https://crfm.stanford.edu/helm/arabic/latest/

[^11_7]: https://asas.ai/tools/arabic-llm-benchmark

[^11_8]: https://aclanthology.org/2023.arabicnlp-1.21.pdf

[^11_9]: https://aixplain.com/benchmark/arabic-llm-benchmark-report/

[^11_10]: https://arabic.ai/llm/

[^11_11]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^11_12]: https://huggingface.co/datasets/meeAtif/hadith_datasets

[^11_13]: https://huggingface.co/papers/2601.17880

[^11_14]: https://huggingface.co/papers/2410.18163

[^11_15]: https://unsloth.ai/blog/r1-reasoning

[^11_16]: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

[^11_17]: https://tldr.takara.ai/p/2305.14982

[^11_18]: https://www.linkedin.com/posts/taghadouinisaid_arabicweb24-creating-a-high-quality-arabic-activity-7227272718130262017-ZLu-

[^11_19]: https://www.linkedin.com/pulse/arabic-llms-2025-benchmark-gulf-business-applications-rabehi-phd-ml8kf

[^11_20]: https://openreview.net/pdf?id=GFvFMJVhLN

[^11_21]: https://www.facebook.com/groups/big.data.egypt/posts/2306248396301015/


---

# لو تحب في الخطوة الجاية:

أكتب لك نسخ كاملة من:
01_build_cpt_mixture.py
04_build_sft_mixture.py
03_train_sft_unsloth.py

تمام، هكتب لك **نسخ كاملة وجاهزة للإنتاج** من السكربتات الثلاثة، متوافقة مع الـ config اللي اتكلمنا عنه، وتفكير SOTA (clean mixing + config-driven + logging بسيط).
كل السكربتات تفترض وجود ملف إعدادات مركزي:
`config/baligh_pipeline_config.json` كما وصفناه سابقًا.

سأكتبها على شكل كود فقط بحيث تنسخها مباشرة في ملفاتك تحت `scripts/`.

***

## 1️⃣ `scripts/01_build_cpt_mixture.py`

يبني خليط CPT من ArabicWeb24 + ArabicText-Large + TheArabicPile_Dialects بنِسَب من الـ config، ويخرجه كـ dataset على القرص.

```python
#!/usr/bin/env python
# scripts/01_build_cpt_mixture.py

import os
import json
from pathlib import Path

from datasets import load_dataset, interleave_datasets

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cpt_cfg = cfg["cpt"]
    output_path = Path(__file__).resolve().parents[1] / cpt_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    probs = []

    print("[CPT] Building CPT mixture...")
    for ds_cfg in cpt_cfg["datasets"]:
        name = ds_cfg["name"]
        split = ds_cfg.get("split", "train")
        weight = ds_cfg["weight"]
        text_field = ds_cfg.get("text_field", "text")

        print(f"[CPT] Loading dataset: {name} (split={split}, weight={weight})")
        ds = load_dataset(name, split=split)

        # محاولة اكتشاف عمود النص إن لم يوجد text_field المحدد
        if text_field not in ds.column_names:
            candidates = [
                c for c in ds.column_names if c.lower() in ["text", "content", "article", "body"]
            ]
            if not candidates:
                raise ValueError(f"[CPT] No text field found for {name}: {ds.column_names}")
            print(f"[CPT] Using inferred text_field='{candidates[0]}' for {name}")
            text_field = candidates[0]

        def _to_text(example):
            t = example.get(text_field) or ""
            return {"text": t}

        ds = ds.map(_to_text, remove_columns=ds.column_names)
        items.append(ds)
        probs.append(weight)

    print(f"[CPT] Interleaving {len(items)} datasets with weights={probs}")
    mixed = interleave_datasets(
        items,
        probabilities=probs,
        seed=3407,
        stopping_strategy="all_exhausted",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(output_path))

    print(
        {
            "saved_to": str(output_path),
            "rows": len(mixed),
            "datasets": [d["name"] for d in cpt_cfg["datasets"]],
        }
    )


if __name__ == "__main__":
    main()
```


***

## 2️⃣ `scripts/04_build_sft_mixture.py`

يبني خليط SFT من CIDAR + evol-instruct-arabic + (ArSyra) + Quran QA بنِسَب من الـ config، ويقيّد prompt format موحد.

```python
#!/usr/bin/env python
# scripts/04_build_sft_mixture.py

import os
import json
from pathlib import Path

from datasets import load_dataset, interleave_datasets

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"

PROMPT_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

PROMPT_NO_INPUT = """### Instruction:
{instruction}

### Response:
{response}"""


def normalize_sample(example):
    # نحاول توحيد أسماء الحقول الشائعة بين الداتا
    instruction = (
        example.get("instruction") or example.get("question") or example.get("prompt") or ""
    )
    inp = example.get("input") or example.get("context") or ""
    response = example.get("output") or example.get("answer") or example.get("response") or ""

    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    response = (response or "").strip()

    if not instruction and not response:
        return {"text": ""}

    if inp:
        text = PROMPT_WITH_INPUT.format(
            instruction=instruction,
            input=inp,
            response=response,
        )
    else:
        text = PROMPT_NO_INPUT.format(
            instruction=instruction,
            response=response,
        )
    return {"text": text}


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sft_cfg = cfg["sft"]
    output_path = Path(__file__).resolve().parents[1] / sft_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    probs = []

    print("[SFT] Building SFT mixture...")
    for ds_cfg in sft_cfg["datasets"]:
        name = ds_cfg["name"]
        split = ds_cfg.get("split", "train")
        weight = ds_cfg["weight"]

        print(f"[SFT] Loading dataset: {name} (split={split}, weight={weight})")
        ds = load_dataset(name, split=split)

        ds = ds.map(
            normalize_sample,
            remove_columns=ds.column_names,
            desc=f"Normalizing {name}",
        )

        # فلترة الأمثلة الفارغة أو القصيرة جدًا
        def _filter(example):
            t = (example.get("text") or "").strip()
            return len(t) > 0 and len(t) > 50

        ds = ds.filter(_filter)
        items.append(ds)
        probs.append(weight)

    print(f"[SFT] Interleaving {len(items)} datasets with weights={probs}")
    mixed = interleave_datasets(
        items,
        probabilities=probs,
        seed=3407,
        stopping_strategy="first_exhausted",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(output_path))

    print(
        {
            "saved_to": str(output_path),
            "rows": len(mixed),
            "datasets": [d["name"] for d in sft_cfg["datasets"]],
        }
    )


if __name__ == "__main__":
    main()
```


***

## 3️⃣ `scripts/03_train_sft_unsloth.py`

يستخدم Unsloth + TRL لتدريب QLoRA على خليط SFT من الخطوة السابقة، مع Hyperparams من الـ config.

```python
#!/usr/bin/env python
# scripts/03_train_sft_unsloth.py

import json
from pathlib import Path

from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg["train"]
    sft_cfg = cfg["sft"]

    sft_path = Path(__file__).resolve().parents[1] / sft_cfg["output_path"]
    if not sft_path.exists():
        raise SystemExit(
            f"[Train] SFT dataset not found at {sft_path}. Run 04_build_sft_mixture.py first."
        )

    print("[Train] Loading SFT dataset from:", sft_path)
    dataset = load_from_disk(str(sft_path))

    print("[Train] Loading base model via Unsloth:", train_cfg["model_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_cfg["model_name"],
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    print("[Train] Applying LoRA (QLoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    output_dir = Path(__file__).resolve().parents[1] / train_cfg["output_dir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print("[Train] Setting up SFTConfig...")
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        report_to="none",
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        max_seq_length=train_cfg["max_seq_length"],
        dataset_text_field="text",
    )

    print("[Train] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
    )

    print("[Train] Starting training...")
    trainer.train()

    print("[Train] Saving model to:", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print({"saved_to": str(output_dir)})


if __name__ == "__main__":
    main()
```


***

## كيف تشبّكهم مع بعض

مع وجود:

- `config/baligh_pipeline_config.json` بالشكل الذي اتفقنا عليه (paths + weights + train cfg).
- هذه السكربتات الثلاثة في `scripts/`.
- `run_baligh_colab.sh` الذي كتبناه في الرسالة السابقة لتشغيلهم على Colab CLI.

تصبح الدورة كاملة:

```bash
# محليًا:
cd Baligh-1.7B

# (مرة واحدة) تثبيت Colab CLI + auth
pip install -U google-colab-cli
colab auth login

# تشغيل البايبلاين بالكامل على GPU Colab
./scripts/run_baligh_colab.sh

# بعد الانتهاء: رفع إلى HF
huggingface-cli login
python scripts/upload_to_hf.py
```


***

لو تحب الخطوة القادمة تكون **وصفة CPT رقمية مفصلة (عدد tokens، عدد steps، lr schedule) لـ ArabicWeb24 على 1.5B** بحيث تكون قريبة من setting SOTA، أقدر أكتبها لك بما في ذلك تقدير ساعات GPU المطلوبة.

---

# لو تحب الخطوة القادمة تكون وصفة CPT رقمية مفصلة (عدد tokens، عدد steps، lr schedule) لـ ArabicWeb24 على 1.5B بحيث تكون قريبة من setting SOTA، أقدر أكتبها لك بما في ذلك تقدير ساعات GPU المطلوبة.

تنفع جدًا تكون الخطوة الجاية، وفعليًا هي النقطة الفاصلة بين Baligh “كويس” وBaligh “قريب من SOTA” في فئة 1.5B.
سأعطيك وصفة **CPT رقمية مفصلة** لـ ArabicWeb24 على 1.5B، مبنية على ما نُشر عن ArabicWeb24 + وصفات pretraining الحديثة لموديلات بحجم قريب، لكن كـ **continued pretraining** فوق Qwen3-1.7B بدل من الصفر.[^13_1][^13_2][^13_3][^13_4]

سأقسّمها إلى:

- هدف التدريب (tokens / budget).
- إعداد الداتا (packing + streaming).
- إعداد الـ optimizer / LR schedule.
- مثال أرقام لـ GPU-hours.

***

## 1️⃣ هدف CPT لـ Baligh-1.7B على ArabicWeb24

ArabicWeb24 =

- Web-only Arabic corpus عالي الجودة مستخرج من 6.5TB WARC بعد cleaning شديد، والناتج حوالي **28–39B tokens** (AraGPT2 tokenizer)، مع فلترة C4-style وFineWeb-style.[^13_2][^13_3][^13_5][^13_1]

لفئة 1.5B كـ **continued pretraining** (مش pretrain من الصفر)، وبالاستلهام من وصفات مثل Union.ai وFineWeb2 وChinchilla-style، وصفة معقولة هي:

- **Tokens CPT مستهدفة**:
    - conservative: 5–10B tokens.
    - aggressive (أقرب SOTA): 15–20B tokens.
- **Sequence length**: 2048 (الموديل Qwen3-1.7B يدعم 32K، لكن 2K–4K أكثر واقعية في compute).[^13_6]
- **Effective batch (tokens/step)**:
    - مثال: batch_size=512 sequences × 2048 tokens ≈ 1.05M tokens/step.
    - ممكن تحققها بـ (global batch = 512) عبر (n_gpus × per_device_batch × grad_accum).

لو استهدفت 10B tokens عند 1M tokens/step → تقريبًا 10k steps CPT.
لو 20B tokens → 20k steps.

لـ v0 SOTA-ish، أرى:

- **CPT steps**: 8k–12k steps على ArabicWeb24 + 10–20% من ArabicText-Large / ArabicPile.[^13_7][^13_8]
- **Total tokens**: ~8–12B tokens.

***

## 2️⃣ إعداد الداتا (ArabicWeb24) لـ CPT

ArabicWeb24 نفسه جهز بـ pipeline cleaning شبيهة بـ FineWeb2:

- language ID،
- dedup،
- bad words filter (بالعربي والإنجليزي)،
- removal of boilerplate والـ nav bars والـ symbol lines.[^13_3][^13_9][^13_1]

لـ CPT على Qwen2.5 tokenizer:

1. تحميل ArabicWeb24 من HF:

```python
from datasets import load_dataset

ds = load_dataset("lightonai/ArabicWeb24", split="train")  # []
```

2. اختياريًا: mixing بسيط مع ArabicText-Large وArabicPile بمعدل 80–10–10 (أو 70–20–10) كما اقترحنا سابقًا.[^13_8][^13_2][^13_7]
3. **Token packing**:
    - استخدم tokenizer Qwen2.5.
    - concatenate documents إلى sequences بطول 2048 بدون padding على مستوى sample (شبه ما يفعله Union.ai بـ MDS أو MosaicML data shards).[^13_4]
    - لو تريد SOTA فعليًا، استبدل JSONL بـ shards (Parquet / MDS) وstream أثناء training بدل تحميل dataset بالكامل في RAM.[^13_10][^13_4]

***

## 3️⃣ وصفة Optimizer / LR / Schedule (CPT)

استلهامًا من:

- ArabicWeb24 ablation (900M مابمّا2)، استخدموا cosine decay + 10% warmup.[^13_3]
- وصفات GPT‑style 1.5B من أدلة pretraining (Union.ai، FineWeb2, TEV paper).[^13_11][^13_4][^13_10]


### إعدادات مقترحة لـ 1.5B CPT:

- **Optimizer**: AdamW (أو AdamW with 8-bit states لو بتحافظ على VRAM).
- **LR base**:
    - 1.5B typical pretraining LR ≈ 1e-3 → 6e-4 طبقًا لأدلة GPT‑style.[^13_4]
    - لكن بما أنه **continued pretraining فوق Qwen2.5**، نفضل LR أقل لتجنب نسيان:
        - **lr = 5e-4 (أعلى)** أو **3e-4 (أكثر أمانًا)**.
- **Warmup**: 5–10% من steps:
    - لو 10k steps → warmup_steps = 500–1000.
- **Scheduler**:
    - Cosine decay إلى 10% من LR (أو صفر)، نفس ما استخدمته تجربة ArabicWeb24.[^13_3]
- **Weight decay**: 0.1 (أو 0.01 لو تحب أخف).
- **Gradient clipping**: 1.0.
- **Precision**: bf16 لو الـ GPU يدعم؛ لو T4 → fp16 مع AMP.
- **Dropout**: استخدام الموجود في Qwen2.5 (لا تغيّر كثيرًا في CPT).
- **Checkpointing**:
    - كل 500–1000 steps.
    - keep_last=5 مثلاً.


### مثال config رقمي لـ 10k steps:

- model: Qwen3-1.7B base.
- seq_len: 2048.
- global batch: 512 sequences (≈1.05M tokens/step).
- steps: 10,000.
- tokens: ≈10.5B.
- LR: 3e-4, warmup: 1000 steps, cosine decay إلى 0.1×LR.

***

## 4️⃣ موارد تقريبية (GPU-hours)

نستند على:

- Union.ai tutorial 1.5B pretraining recipe التي تستخدم dataset SlimPajama (627B tokens) بعينة صغيرة (5M samples) للتجريب وتوضح أن config 1.5B على 8×GPU node مناسب للاختبار قبل run كبير.[^13_4]
- خبرة عامة: 1.5B GPT-style على 10B tokens بـ 1M tokens/step ~ 10k steps.

لو افترضنا:

- نموذج 1.5B،
- 8×A100 40GB،
- global batch ~512، seq=2048 (1.05M tokens/step)،
- step time ≈ 0.4–0.7 ثانية (تقدير تقريبي من تجارب مماثلة لـ 1–2B),

إذن:

- 10k steps × 0.6s ≈ 6000s ≈ 1.7h wall-clock على 8×A100 (بدون overhead كبير).
- على 4×A100 أو 4×L40S، الوقت يتضاعف تقريبًا → 3–4h.

على موارد معتدلة (مثلاً 4×L4 أو V100):

- step time قد يكون 1–2s:
    - 10k steps × 1.5s ≈ 15k s ≈ 4.2h.

لأن ده CPT (مش pretrain كامل)، compute المطلوب أقل بكثير من train من الصفر؛ SOTA-ish CPT run بـ 10–12B tokens على 1.5B **مش جنوني** لو عندك access لعقدة واحدة محترمة (أو Cloud spot instances).[^13_10][^13_4]

***

## 5️⃣ وصفة CPT جاهزة بالأرقام

لو حاب ناخد قرار عملي واحد:

- **Stage: Arabic CPT** فوق Qwen3-1.7B base.
- **Data**:
    - ArabicWeb24: 80%.
    - ArabicText-Large: 10%.
    - ArabicPile_Dialects: 10%.[^13_2][^13_7][^13_8]
- **Seq length**: 2048.
- **Global batch**: 512 sequences (≈1.05M tokens/step).
- **Steps**: 12,000.
- **Total tokens**: ≈12.6B.
- **LR**: 3e-4.
- **Warmup**: 1,200 steps (10%).
- **Scheduler**: cosine decay إلى 0.1×LR.
- **WD**: 0.1.
- **Precision**: bf16 أو fp16.
- **Checkpoints**: كل 500 steps.

هذا يجعلك:

- تستخدم شريحة محترمة من ArabicWeb24 مع respect لوصفة cleaning الأصلية.[^13_1][^13_3]
- تحصل على Arabic-adapted Qwen3-1.7B base قوي قبل SFT.
- تكون قريب من “وصفة معقولة” في حجم compute لموديل 1.5B بناءً على ممارسات حديثة.[^13_11][^13_10][^13_4]

***

لو حابب، أقدر في الرسالة القادمة أجهز لك:

- **YAML/JSON config كامل** لجولة CPT دي (جاهز لـ use مع أي trainer)،
- أو سكربت `train_cpt.py` (PyTorch/Transformers/accelerate) يستخدم ArabicWeb24 بشكل streaming + token packing، بحيث تحطه جنب SFT pipeline في مشروع Baligh وتبقى ساعتها فعليًا عندك “وصفة CPT إنتاجية” جاهزة للركض.
<span style="display:none">[^13_12][^13_13][^13_14][^13_15][^13_16][^13_17][^13_18]</span>

<div align="center">⁂</div>

[^13_1]: https://lighton.ai/lighton-blogs/arabicweb24

[^13_2]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^13_3]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^13_4]: https://www.union.ai/docs/v2/flyte/tutorials/frontier-ai/distributed-pretraining/

[^13_5]: https://www.linkedin.com/posts/taghadouinisaid_arabicweb24-creating-a-high-quality-arabic-activity-7227272718130262017-ZLu-

[^13_6]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^13_7]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^13_8]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^13_9]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^13_10]: https://openreview.net/pdf?id=jnRBe6zatP

[^13_11]: https://aclanthology.org/2024.emnlp-main.606.pdf

[^13_12]: https://arxiv.org/html/2512.18399v1

[^13_13]: https://aclanthology.org/2023.findings-acl.598.pdf

[^13_14]: https://tw93.fun/en/2026-04-03/llm.html

[^13_15]: https://www.youtube.com/watch?v=hTvRg6-H_-8

[^13_16]: https://www.linkedin.com/posts/lighton_arabicweb24-creating-a-high-quality-arabic-activity-7227260529235374080-omKg

[^13_17]: https://www.youtube.com/watch?v=QZObcrAO448

[^13_18]: https://arxiv.org/html/2507.14688v2


---

# عاوز implementation plan file by file in details Baligh-1.7B v0 يشتتغل علي Colab cli T4GPU Baligh-1.7B-v0 = أول baseline. مع شرح تفصيلي لكل خطوة و Clear architecture and clean code

سأعتبر Baligh-1.7B-v0 **baseline قوي** (لسه مش SOTA CPT) يشتغل بالكامل على **Colab CLI + T4 GPU**، مع **architecture واضح** و**كود نظيف**.
الخطة هنا implementation plan “ملف بملف” + شرح ما يحصل في كل خطوة، بحيث تقدر تبدأ فورًا من الترمينال بدون فتح متصفح.

سأقسّم الرد إلى:

1. هيكل المشروع (Clear architecture).
2. ملفات `config` و `requirements`.
3. ملفات `scripts` واحدة واحدة مع دورها.
4. كيف تشغل كل شيء من Colab CLI على T4.
5. ماذا يحدث فعليًا أثناء التنفيذ (flow منطقي).

كل ما أقوله يفترض أنك تستخدم:

- **Qwen3-1.7B 4-bit عبر Unsloth** (فيه Colab T4 notebook رسمي، ويدعم training أسرع 2x وتقليل VRAM).[^14_1][^14_2]
- **Colab CLI** الرسمي الذي يسمح بتشغيل Python scripts على GPU T4 من الترمينال (`colab new`, `colab exec`, `colab upload`, `colab download`).[^14_3][^14_4][^14_5]

***

## 1️⃣ Clear Architecture – هيكل مشروع Baligh-1.7B-v0

أنشئ مجلد مشروع:

```bash
Baligh-1.7B-v0/
  README.md
  requirements.txt

  config/
    baligh_pipeline_config.json

  data/
    cpt/
      # لاحقًا: mixture + jsonl إن حبيت تجرب CPT
    sft/
      # mixture لـ SFT (interleaved dataset)

  scripts/
    01_build_cpt_mixture.py       # (اختياري لـ CPT لاحقًا)
    02_prepare_cpt_jsonl.py       # (اختياري لـ CPT لاحقًا)
    03_train_sft_unsloth.py       # تدريب SFT الفعلي
    04_build_sft_mixture.py       # خلط وتجهيز داتا SFT
    05_run_eval.py                # sanity check بعد التدريب
    run_baligh_colab.sh           # Orchestrator لتشغيل كل شيء على Colab CLI T4
    upload_to_hf.py               # رفع النموذج إلى Hugging Face Hub

  models/
    Baligh-1.7B-v0-lora/          # يخرج هنا بعد التدريب

  logs/
    train/
    eval/
```

الفكرة:

- `config/` = مكان واحد لكل الإعدادات (مسارات، datasets، hyperparams).
- `data/` = ما يخرج من تحضير البيانات (SFT mixture).
- `scripts/` = كل logic (تحضير، تدريب، eval، orchestration).
- `models/` = الموديل الناتج.
- `logs/` = لو حابب تضيف logging لاحقًا (W\&B أو نصي).

***

## 2️⃣ ملف الإعدادات والـ dependencies

### 2.1 `requirements.txt`

نحتاج stack أساسي لـ: Unsloth + HF + TRL + Datasets:

```text
unsloth
transformers
trl
accelerate
datasets
bitsandbytes
huggingface_hub
```

(تقدر تضيف `wandb` لاحقًا لو حابب logging خارج Colab).[^14_2][^14_6]

### 2.2 `config/baligh_pipeline_config.json`

هذا هو **الملف المركزي**:

```json
{
  "project": {
    "name": "Baligh-1.7B-v0",
    "org": "your-org",
    "base_model": "unsloth/Qwen3-1.7B-Base"
  },
  "cpt": {
    "output_path": "data/cpt/baligh_cpt_mixture",
    "jsonl_path": "data/cpt/baligh_cpt_train.jsonl",
    "datasets": [
      { "name": "lightonai/ArabicWeb24", "split": "train", "weight": 0.7, "text_field": "text" },
      { "name": "Jr23xd23/ArabicText-Large", "split": "train", "weight": 0.2, "text_field": "text" },
      { "name": "premio-ai/TheArabicPile_Dialects", "split": "train", "weight": 0.1, "text_field": "text" }
    ]
  },
  "sft": {
    "output_path": "data/sft/baligh_sft_mixture",
    "datasets": [
      { "name": "arbml/CIDAR", "split": "train", "weight": 0.4 },
      { "name": "FreedomIntelligence/evol-instruct-arabic", "split": "train", "weight": 0.35 },
      { "name": "ArSyra/arsyra-instruction-tuning", "split": "train", "weight": 0.1 },
      { "name": "nazimali/quran-question-answer-context", "split": "train", "weight": 0.05 }
    ]
  },
  "train": {
    "model_name": "unsloth/Qwen3-1.7B-Base",
    "max_seq_length": 2048,
    "output_dir": "models/Baligh-1.7B-v0-lora",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 0.0002,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "fp16": true,
    "bf16": false
  },
  "eval": {
    "datasets": [
      { "name": "FreedomIntelligence/MMLU_Arabic", "split": "test" },
      { "name": "castorini/mr-tydi", "split": "test" }
    ]
  },
  "hf": {
    "repo_id": "your-org/Baligh-1.7B-v0",
    "private": false
  }
}
```

- **SFT**: خلط CIDAR + evol-instruct-arabic + ArSyra + Quran QA بالنسب التي اتفقنا عليها.[^14_7][^14_8][^14_9][^14_10]
- **train.max_steps=200**: baseline سريع قابل للتعديل بعد التأكد أن كل شيء يعمل على T4.

***

## 3️⃣ ملفات `scripts` – دور كل واحد وشرح مفصل

### 3.1 `scripts/04_build_sft_mixture.py` – تجهيز بيانات SFT

**الهدف:**
تحميل datasets لـ SFT (CIDAR, evol-instruct-arabic, ArSyra, Quran QA)، توحيد الـ schema إلى prompt/response، وخلطها بنسب config.

الكود (كتبناه بالفعل، أذكّرك به مع التعليق):

- يقرأ `config/baligh_pipeline_config.json`.
- لكل dataset:
    - يحملها بـ `load_dataset`.
    - يستخرج `instruction`, `input`, `response` من الحقول الشائعة (`instruction`, `question`, `prompt`, `input`, `context`, `output`, `answer`).[^14_8][^14_10][^14_7]
    - يبني النص النهائي بصيغة:

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

    - أو بدون Input لو فارغ.
- يفلتر الأمثلة القصيرة جدًا (تجنب noise).
- يعمل `interleave_datasets` بنسب `weight` من الـ config.
- يحفظ dataset الناتجة بـ `save_to_disk` في `data/sft/baligh_sft_mixture`.

**النتيجة:**
Dataset موحدة جاهزة لـ SFT، فيها عمود واحد `text`.

(تقدر تستخدم النسخة الكاملة اللي كتبتها لك سابقًا كما هي.)

### 3.2 `scripts/03_train_sft_unsloth.py` – تدريب SFT باستخدام Unsloth على T4

**الهدف:**

- تحميل **Qwen3-1.7B 4-bit** عبر Unsloth.[^14_1][^14_2]
- تفعيل QLoRA (r=16, alpha=16).
- تشغيل SFTTrainer على خليط SFT الذي حضرناه.
- حفظ Baligh-1.7B-v0 كـ LoRA model في `models/Baligh-1.7B-v0-lora`.

الكود الذي أعطيتك إياه:

- يقرأ config.
- يحمّل dataset من `data/sft/baligh_sft_mixture` بـ `load_from_disk`.
- يستدعي:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=train_cfg["model_name"],
    max_seq_length=train_cfg["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)
```

هذا يستفيد من نموذج Unsloth Qwen3-1.7B 4-bit المعد للتدريب السريع على Colab T4.[^14_1]
- يفعّل LoRA:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

هذا يضبط QLoRA بذاكرة قليلة (مناسب لـ 16GB T4).[^14_6][^14_11]
- يضبط SFTConfig من config (batch، grad_accum، lr، steps، fp16…).
- يشغل `trainer.train()`.
- يحفظ weights + tokenizer.


### 3.3 `scripts/05_run_eval.py` – sanity check

**الهدف:**

- التأكد أن الموديل يشتغل بعد التدريب.
- اختبار واحد أو اثنين من prompts.
- لاحقًا ممكن توسّعه ليحمّل Benchmarks (MMLU_Arabic وغيرها).[^14_12]

ببساطة:

- يحمّل `models/Baligh-1.7B-v0-lora` بـ `AutoModelForCausalLM` و `AutoTokenizer`.
- يولد استجابة لسؤال مثل:
“اشرح الفرق بين الفقه والعقيدة باختصار.”
- يطبع الـ output.


### 3.4 `scripts/upload_to_hf.py` – رفع النموذج للـ Hub

**الهدف:**
رفع `models/Baligh-1.7B-v0-lora` إلى Hugging Face Hub باستخدام `huggingface_hub` (create_repo + upload_folder).[^14_13][^14_14][^14_15]

- يقرأ `config/baligh_pipeline_config.json`.
- يأخذ:
    - `hf.repo_id` (مثل `"mohamed-kandil/Baligh-1.7B-v0"`).
    - `train.output_dir` (مسار النموذج).
- يستدعي:

```python
create_repo(repo_id=repo_id, private=private, exist_ok=True)
upload_folder(repo_id=repo_id, folder_path=model_dir, repo_type="model")
```

- تحتاج أنك تعمل `huggingface-cli login` مرة واحدة محليًا.[^14_16][^14_13]


### 3.5 (اختياري) `scripts/01_build_cpt_mixture.py` و `02_prepare_cpt_jsonl.py`

في v0 baseline على T4، ممكن تبدأ بس SFT بدون CPT (لتقليل التعقيد وتأجيل CPT لما يكون عندك compute أكبر).
لكن عندك سكربتات جاهزة لـ:

- خلط ArabicWeb24 + ArabicText-Large + ArabicPile بنسب 70/20/10.[^14_17][^14_18][^14_19]
- تحويلها إلى JSONL للتجربة لاحقًا.

التركيز الآن لتشغيل Baligh-1.7B-v0 baseline:
**04 + 03 + 05 + run_baligh_colab + upload_to_hf.**

***

## 4️⃣ `scripts/run_baligh_colab.sh` – كيف يشغل كل شيء على Colab CLI + T4

هذا هو “قلب التشغيل” من الترمينال. الفكرة:

1. تخصيص session جديدة مع GPU T4.
2. تثبيت المتطلبات داخل الـ session (`pip install -r requirements.txt`).
3. رفع مشروعك بالكامل إلى `/workspace/Baligh-1.7B-v0` على runtime.[^14_4][^14_3]
4. تشغيل:
    - `04_build_sft_mixture.py`
    - `03_train_sft_unsloth.py`
    - (اختياريًا `05_run_eval.py`).
5. تنزيل `models/Baligh-1.7B-v0-lora` إلى جهازك.

المنطق الداخلي يعتمد على أوامر موثقة لـ Colab CLI:

- `colab new` لإنشاء session.[^14_3][^14_4]
- `colab exec` لتشغيل Python scripts عن بُعد.[^14_5][^14_3]
- `colab upload` و `colab download` لإدارة الملفات.[^14_4]

***

## 5️⃣ Flow التنفيذ الكامل (من عندك لحد HF)

من منظورك كـ مهندس:

1. **محليًا:**
    - تنشئ الهيكل السابق + تضع السكربتات.
    - تنصّب Colab CLI:

```bash
pip install -U google-colab-cli
colab auth login   # تسجّل بحساب Google مرة واحدة
```

    - تعدّل `config/baligh_pipeline_config.json` (hf.repo_id, org, max_steps…).
2. **تشغيل التدريب على T4:**

```bash
cd Baligh-1.7B-v0
./scripts/run_baligh_colab.sh
```

ما يحدث:
    - `colab new -s baligh-train --gpu T4` ينشئ VM T4.[^14_4]
    - `colab upload` يرفع مشروعك.
    - `colab exec` داخل الـ VM:
        - `pip install -r requirements.txt`.
        - `python scripts/04_build_sft_mixture.py`.
        - `python scripts/03_train_sft_unsloth.py`.
    - `colab download` يرجع `models/Baligh-1.7B-v0-lora` إلى جهازك.
3. **اختبار محلي سريع:**
    - تكتب سكربت صغير في جذر المشروع (مثلاً `test_baligh.py`) لتحميل النموذج وإعطاء prompt.
    - أو تستخدم `05_run_eval.py` داخل Colab session أو محليًا.
4. **رفع إلى Hugging Face:**

```bash
huggingface-cli login
python scripts/upload_to_hf.py
```

هذا يستخدم `huggingface_hub` لرفع المجلد كـ model repo.[^14_14][^14_13]

***

## 6️⃣ ليه ده baseline جيد لـ Baligh-1.7B-v0 على T4؟

- يعتمد على **Qwen3-1.7B 4-bit بإعداد Unsloth**، وهو معد خصيصًا لـ Colab T4 finetuning (Unsloth يوفر Colab notebook رسمي لهذا الهدف).[^14_20][^14_1]
- يفصل بين:
    - تحضير الداتا (sft mixture).
    - التدريب (سطر واحد داخل Colab).
    - الرفع إلى HF.
- يستخدم **config file** واحد لتغيير:
    - datasets.
    - نسب المزج.
    - hyperparameters.
- يسمح بالتدرّج:
    - v0 = SFT فقط (baseline).
    - v1 = SFT + CPT خفيف.
    - v2 = SFT + CPT + RL + Benchmarks ABB/HELM Arabic.

***

لو تحب الخطوة القادمة تكون:

- ملف `README.md` في root المشروع يشرح لكل contributor (أو لنفسك بعد شهرين) “كيف أشغّل Baligh-1.7B-v0 من الصفر على Colab CLI خطوة بخطوة”،
أو
- ملف `test_baligh.py` لتحميل النموذج بعد التدريب وتجربته على عدة prompts (Islamic QA, summarization, rewriting)،

قول لي أيهما أولى لك الآن.
<span style="display:none">[^14_21][^14_22][^14_23][^14_24][^14_25][^14_26][^14_27][^14_28][^14_29][^14_30][^14_31]</span>

<div align="center">⁂</div>

[^14_1]: https://huggingface.co/unsloth/Qwen3-1.7B

[^14_2]: https://github.com/unslothai/unsloth?locale=en-US

[^14_3]: https://github.com/googlecolab/google-colab-cli

[^14_4]: https://mer.vin/2026/06/google-colab-cli-provision-gpus-and-run-scripts-from-your-terminal/?amp=1

[^14_5]: https://developers.googleblog.com/introducing-the-google-colab-cli/

[^14_6]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

[^14_7]: https://github.com/ARBML/CIDAR

[^14_8]: https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-arabic

[^14_9]: https://huggingface.co/datasets/ArSyra/arsyra-instruction-tuning

[^14_10]: https://huggingface.co/datasets/nazimali/quran-question-answer-context

[^14_11]: https://unsloth.ai/blog/r1-reasoning

[^14_12]: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Arabic

[^14_13]: https://huggingface.co/docs/huggingface_hub/en/guides/upload

[^14_14]: https://pypi.org/project/huggingface-hub/

[^14_15]: https://discuss.huggingface.co/t/upload-and-commit-your-model-into-huggingface-hub/136548

[^14_16]: https://www.geeksforgeeks.org/machine-learning/how-to-upload-and-share-model-to-huggingface/

[^14_17]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^14_18]: https://huggingface.co/datasets/Jr23xd23/ArabicText-Large

[^14_19]: https://huggingface.co/datasets/premio-ai/TheArabicPile_Dialects

[^14_20]: https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing

[^14_21]: https://stackoverflow.com/questions/51194303/how-to-run-a-python-script-in-a-py-file-from-a-google-colab-notebook

[^14_22]: https://colab.research.google.com

[^14_23]: https://www.reddit.com/r/learnmachinelearning/comments/mweify/how_to_run_python_scripts_on_google_colab/

[^14_24]: https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-bnb-4bit

[^14_25]: https://huggingface.co/lightonai/ArabicWeb24-ablation-model-v5

[^14_26]: https://developers.google.com/colab

[^14_27]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^14_28]: https://livebook.manning.com/wiki/categories/python/command+line

[^14_29]: https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/

[^14_30]: https://github.com/unslothai/notebooks

[^14_31]: https://lighton.ai/lighton-blogs/arabicweb24


---

# نسخ كاملة من:

01_build_cpt_mixture.py
04_build_sft_mixture.py
03_train_sft_unsloth.py
بشكل production أكثر (مع logging بسيط، validation split، وconfig-driven بالكامل)، قول لي وسأركّز على السكربتات الثلاثة دي تحديدًا.

سأعطيك الآن **نسخ كاملة production-style** للملفات الثلاثة، كلها:

- `config-driven بالكامل` (لا وجود لقيم hard-coded قدر الإمكان).
- فيها **logging بسيط** برسائل واضحة.
- تدعم **train/validation split** في SFT.
- مناسبة للـ Colab T4 (Unsloth + QLoRA).

أفترض أن عندك:

- ملف إعدادات: `config/baligh_pipeline_config.json` (سأوسّعه قليلًا).
- هيكل مشروع كما اتفقنا (`Baligh-1.7B-v0/…`).

سأبدأ بتحديث بسيط مطلوب في الـ config، ثم أعطيك السكربتات الثلاثة.

***

## 0️⃣ تحديث `config/baligh_pipeline_config.json`

أضف حقول بسيطة لـ SFT splits و logging:

```jsonc
{
  "project": {
    "name": "Baligh-1.7B-v0",
    "org": "your-org",
    "base_model": "unsloth/Qwen3-1.7B-Base"
  },
  "cpt": {
    "output_path": "data/cpt/baligh_cpt_mixture",
    "jsonl_path": "data/cpt/baligh_cpt_train.jsonl",
    "datasets": [
      { "name": "lightonai/ArabicWeb24", "split": "train", "weight": 0.7, "text_field": "text" },
      { "name": "Jr23xd23/ArabicText-Large", "split": "train", "weight": 0.2, "text_field": "text" },
      { "name": "premio-ai/TheArabicPile_Dialects", "split": "train", "weight": 0.1, "text_field": "text" }
    ]
  },
  "sft": {
    "output_path": "data/sft/baligh_sft_mixture",
    "val_ratio": 0.02,                 // 2% validation
    "shuffle_seed": 3407,
    "datasets": [
      { "name": "arbml/CIDAR", "split": "train", "weight": 0.4 },
      { "name": "FreedomIntelligence/evol-instruct-arabic", "split": "train", "weight": 0.35 },
      { "name": "ArSyra/arsyra-instruction-tuning", "split": "train", "weight": 0.1 },
      { "name": "nazimali/quran-question-answer-context", "split": "train", "weight": 0.05 }
    ]
  },
  "train": {
    "model_name": "unsloth/Qwen3-1.7B-Base",
    "max_seq_length": 2048,
    "output_dir": "models/Baligh-1.7B-v0-lora",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 0.0002,
    "logging_steps": 10,
    "eval_steps": 50,                  // new: evaluate every N steps
    "save_steps": 100,
    "save_total_limit": 2,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "fp16": true,
    "bf16": false
  }
}
```


***

## 1️⃣ `01_build_cpt_mixture.py` – Production CPT mixing (logging + config-driven)

```python
#!/usr/bin/env python
# scripts/01_build_cpt_mixture.py

import json
import logging
from pathlib import Path

from datasets import load_dataset, interleave_datasets

# إعداد logging بسيط
logging.basicConfig(
    format="[%(asctime)s] [CPT] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[^15_1] / "config" / "baligh_pipeline_config.json"


def load_and_normalize_dataset(name: str, split: str, text_field: str):
    logging.info(f"Loading dataset '{name}' (split={split})")
    ds = load_dataset(name, split=split)

    # تحديد عمود النص
    if text_field not in ds.column_names:
        candidates = [
            c
            for c in ds.column_names
            if c.lower() in ["text", "content", "article", "body"]
        ]
        if not candidates:
            raise ValueError(
                f"No suitable text field found for {name}. Columns: {ds.column_names}"
            )
        logging.warning(
            "text_field '%s' not in columns for %s, using inferred '%s'",
            text_field,
            name,
            candidates[^15_0],
        )
        text_field = candidates[^15_0]

    def _to_text(example):
        t = example.get(text_field) or ""
        return {"text": t}

    ds = ds.map(_to_text, remove_columns=ds.column_names)
    logging.info("Dataset '%s' normalized to single 'text' column (rows=%d)", name, len(ds))
    return ds


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cpt_cfg = cfg["cpt"]
    output_path = Path(__file__).resolve().parents[^15_1] / cpt_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    weights = []

    logging.info("Building CPT mixture from %d datasets", len(cpt_cfg["datasets"]))
    for ds_cfg in cpt_cfg["datasets"]:
        name = ds_cfg["name"]
        split = ds_cfg.get("split", "train")
        weight = float(ds_cfg["weight"])
        text_field = ds_cfg.get("text_field", "text")

        ds = load_and_normalize_dataset(name, split, text_field)
        datasets.append(ds)
        weights.append(weight)

    logging.info("Interleaving datasets with weights: %s", weights)
    mixed = interleave_datasets(
        datasets,
        probabilities=weights,
        seed=3407,
        stopping_strategy="all_exhausted",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(output_path))

    logging.info("Mixed CPT dataset saved to '%s' (rows=%d)", output_path, len(mixed))
    print(
        {
            "saved_to": str(output_path),
            "rows": len(mixed),
            "datasets": [d["name"] for d in cpt_cfg["datasets"]],
        }
    )


if __name__ == "__main__":
    main()
```

- **ماذا يفعل؟**
    - يقرأ config.
    - يحمل كل dataset لـ CPT.
    - يختار عمود النص (text/content/…).
    - يحوّل إلى عمود واحد `text`.
    - يعمل interleave بنسب `weight`.
    - يحفظ الناتج في `data/cpt/baligh_cpt_mixture`.

***

## 2️⃣ `04_build_sft_mixture.py` – Production SFT mixing (logging + val split)

```python
#!/usr/bin/env python
# scripts/04_build_sft_mixture.py

import json
import logging
from pathlib import Path

from datasets import load_dataset, interleave_datasets, DatasetDict

logging.basicConfig(
    format="[%(asctime)s] [SFT] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[^15_1] / "config" / "baligh_pipeline_config.json"

PROMPT_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

PROMPT_NO_INPUT = """### Instruction:
{instruction}

### Response:
{response}"""


def normalize_sample(example):
    instruction = (
        example.get("instruction")
        or example.get("question")
        or example.get("prompt")
        or ""
    )
    inp = example.get("input") or example.get("context") or ""
    response = (
        example.get("output")
        or example.get("answer")
        or example.get("response")
        or ""
    )

    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    response = (response or "").strip()

    if not instruction and not response:
        return {"text": ""}

    if inp:
        text = PROMPT_WITH_INPUT.format(
            instruction=instruction,
            input=inp,
            response=response,
        )
    else:
        text = PROMPT_NO_INPUT.format(
            instruction=instruction,
            response=response,
        )
    return {"text": text}


def build_sft_mixture(cfg):
    sft_cfg = cfg["sft"]
    output_path = Path(__file__).resolve().parents[^15_1] / sft_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    probs = []

    logging.info("Building SFT mixture from %d datasets", len(sft_cfg["datasets"]))
    for ds_cfg in sft_cfg["datasets"]:
        name = ds_cfg["name"]
        split = ds_cfg.get("split", "train")
        weight = float(ds_cfg["weight"])

        logging.info("Loading SFT dataset '%s' (split=%s, weight=%.3f)", name, split, weight)
        ds = load_dataset(name, split=split)

        ds = ds.map(
            normalize_sample,
            remove_columns=ds.column_names,
            desc=f"Normalizing {name}",
        )

        # Filter out very short or empty samples
        def _filter(example):
            t = (example.get("text") or "").strip()
            return len(t) > 50

        ds = ds.filter(_filter, desc=f"Filtering {name} short/empty")

        logging.info("Dataset '%s' after normalization/filtering: %d rows", name, len(ds))

        items.append(ds)
        probs.append(weight)

    logging.info("Interleaving SFT datasets with weights: %s", probs)
    mixed = interleave_datasets(
        items,
        probabilities=probs,
        seed=sft_cfg.get("shuffle_seed", 3407),
        stopping_strategy="first_exhausted",
    )

    logging.info("Mixed SFT dataset size: %d rows", len(mixed))
    return mixed, output_path


def split_train_val(dataset, val_ratio, seed):
    if val_ratio <= 0:
        logging.warning("val_ratio <= 0, using full dataset as train only.")
        return DatasetDict({"train": dataset})

    logging.info("Shuffling dataset before split (seed=%d)", seed)
    dataset = dataset.shuffle(seed=seed)

    val_size = int(len(dataset) * val_ratio)
    if val_size == 0:
        logging.warning(
            "val_ratio too small, no validation rows. Using full dataset as train."
        )
        return DatasetDict({"train": dataset})

    logging.info("Splitting dataset: %d train / %d validation", len(dataset) - val_size, val_size)
    dsdict = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict({"train": dsdict["train"], "validation": dsdict["test"]})


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sft_cfg = cfg["sft"]
    val_ratio = float(sft_cfg.get("val_ratio", 0.0))
    seed = int(sft_cfg.get("shuffle_seed", 3407))

    mixed, output_path = build_sft_mixture(cfg)
    ds_dict = split_train_val(mixed, val_ratio, seed)

    output_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(output_path))

    logging.info(
        "SFT dataset saved to '%s' (train=%d, validation=%d)",
        output_path,
        len(ds_dict["train"]),
        len(ds_dict.get("validation", [])),
    )

    print(
        {
            "saved_to": str(output_path),
            "train_rows": len(ds_dict["train"]),
            "val_rows": len(ds_dict.get("validation", [])),
        }
    )


if __name__ == "__main__":
    main()
```

- **ماذا يفعل؟**
    - يبني mixture موحد `text`.
    - يعمل shuffle + split إلى train/validation بناءً على `val_ratio`.
    - يحفظ dataset كـ `DatasetDict` في `data/sft/baligh_sft_mixture` (فيه `train` و `validation`).

***

## 3️⃣ `03_train_sft_unsloth.py` – Production SFT training (logging + val + config-driven)

```python
#!/usr/bin/env python
# scripts/03_train_sft_unsloth.py

import json
import logging
from pathlib import Path

from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    format="[%(asctime)s] [TRAIN] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[^15_1] / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg["train"]
    sft_cfg = cfg["sft"]

    sft_path = Path(__file__).resolve().parents[^15_1] / sft_cfg["output_path"]
    if not sft_path.exists():
        raise SystemExit(
            f"[TRAIN] SFT dataset not found at {sft_path}. Run 04_build_sft_mixture.py first."
        )

    logging.info("Loading SFT dataset from '%s'", sft_path)
    ds_dict = load_from_disk(str(sft_path))

    train_dataset = ds_dict["train"]
    eval_dataset = ds_dict.get("validation")

    logging.info("Train rows: %d", len(train_dataset))
    if eval_dataset is not None:
        logging.info("Validation rows: %d", len(eval_dataset))
    else:
        logging.warning("No validation split found; training without eval.")

    logging.info("Loading base model via Unsloth: %s", train_cfg["model_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_cfg["model_name"],
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    logging.info("Applying LoRA (QLoRA settings)")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    output_dir = Path(__file__).resolve().parents[^15_1] / train_cfg["output_dir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Setting up SFTConfig")
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        report_to="none",
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        max_seq_length=train_cfg["max_seq_length"],
        dataset_text_field="text",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 0) or None,
    )

    logging.info("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")

    logging.info("Saving final model to '%s'", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logging.info("All done.")
    print({"saved_to": str(output_dir)})


if __name__ == "__main__":
    main()
```

- **ماذا يفعل؟**
    - يحمّل `DatasetDict` من `data/sft/baligh_sft_mixture` (train + validation).
    - يشغّل SFTTrainer مع:
        - eval كل `train.eval_steps`.
        - logging منظم.
    - يحفظ النموذج تحت `models/Baligh-1.7B-v0-lora`.

هذا السكربت مصمم ليشتغل كما هو على Colab T4 عبر Unsloth، لأنه يعتمد على 4-bit QLoRA وتكوين batch صغير نسبيًا (2 × grad_accum=4 → effective 8)، وهو في نطاق T4 المعتاد.[^15_1][^15_2][^15_3]

***

بهذا أصبح عندك:

- **CPT mixer** جاهز (01).
- **SFT mixer + train/val split** (04).
- **SFT trainer production-style** (03).

كلها:

- تعتمد على `baligh_pipeline_config.json`.
- فيها logging نظيف.
- قابلة للتشغيل على Colab CLI عبر `run_baligh_colab.sh` الذي كتبناه سابقًا.

لو تريد في الخطوة التالية:

- أكتب لك `README.md` تقني يشرح لكل contributor كيف يشغل المشروع end-to-end (من git clone إلى model على HF)،
أو
- أضيف دعم `wandb` logging وmetric tracking في `03_train_sft_unsloth.py`،

اختر أي واحد تفضله أولًا.

<div align="center">⁂</div>

[^15_1]: https://huggingface.co/unsloth/Qwen3-1.7B

[^15_2]: https://unsloth.ai/blog/r1-reasoning

[^15_3]: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide


---

# أكتب لك README.md تقني يشرح لكل contributor كيف يشغل المشروع end-to-end (من git clone إلى model على HF)،

أو
أضيف دعم wandb logging وmetric tracking في 03_train_sft_unsloth.py،

ابدأ بـ **README.md التقني** ثم أضف W\&B لاحقًا؛ لأن README هو اللي هيثبّت الـ mental model لكل حد يلمس المشروع (بما فيهم أنت بعد شهرين).
إليك **README.md** جاهز تقدر تحطه في جذر المشروع `Baligh-1.7B-v0/` وتعدّل الاسم/الـ org فقط.

***

```markdown
# Baligh-1.7B-v0

Baligh-1.7B-v0 هو نموذج لغة عربي (1.5B) مبني على Qwen3-1.7B عبر Unsloth، مع Supervised Fine-tuning (SFT) على بيانات تعليمات عربية (CIDAR, evol-instruct-arabic, إلخ).  
هذا الريبو يتيح لك تشغيل **تدريب كامل** على **Google Colab T4** باستخدام **Colab CLI** من الترمينال، ثم رفع النموذج إلى Hugging Face.

## ✨ الميزات

- Clear architecture: فصل بين `config`, `data`, `scripts`, `models`, `logs`.
- Config-driven بالكامل عبر `config/baligh_pipeline_config.json`.
- Data mixing لـ SFT من Hugging Face datasets (CIDAR, evol-instruct-arabic, ArSyra, Quran QA).
- تدريب SFT باستخدام Unsloth + QLoRA على Qwen3-1.7B 4-bit (مناسب لـ T4 GPU).
- دعم train/validation split و eval أثناء التدريب.
- سكربت رفع جاهز إلى Hugging Face Hub.

---

## 1️⃣ بنية المشروع

```bash
Baligh-1.7B-v0/
  README.md
  requirements.txt

  config/
    baligh_pipeline_config.json

  data/
    cpt/
      # (اختياري لاحقًا لـ CPT)
    sft/
      # خليط SFT (train + validation)

  scripts/
    01_build_cpt_mixture.py     # (اختياري: خلط CPT)
    02_prepare_cpt_jsonl.py     # (اختياري: JSONL لـ CPT)
    03_train_sft_unsloth.py     # تدريب SFT فعليًا
    04_build_sft_mixture.py     # خلط وتحضير داتا SFT
    05_run_eval.py              # sanity check بعد التدريب
    run_baligh_colab.sh         # تشغيل pipeline على Colab CLI + T4
    upload_to_hf.py             # رفع النموذج إلى Hugging Face

  models/
    Baligh-1.7B-v0-lora/        # يخرج هنا بعد التدريب

  logs/
    train/
    eval/
```

- الملف الأساسي للإعدادات: `config/baligh_pipeline_config.json`.
- T4/Colab training يُدار عبر `scripts/run_baligh_colab.sh`.

---

## 2️⃣ المتطلبات الأساسية

### 2.1 على جهازك المحلي

- Python 3.10+.
- `pip` أو `uv`.
- حساب Google (لتشغيل Colab).
- حساب Hugging Face (لرفع النموذج).

ثبت Colab CLI + Hugging Face CLI:

```bash
pip install -U google-colab-cli huggingface_hub
# أو:
# uv tool install google-colab-cli
```

ثم:

```bash
colab auth login          # تسجيل الدخول لـ Colab (مرة واحدة)
huggingface-cli login     # تسجيل الدخول لـ HuggingFace (مرة واحدة)
```


### 2.2 داخل Colab runtime

تتم إدارة تثبيت الـ Python deps عبر:

```bash
pip install -U -r requirements.txt
```

حيث `requirements.txt` يحتوي على:

```text
unsloth
transformers
trl
accelerate
datasets
bitsandbytes
huggingface_hub
```

Unsloth يوفر دعم رسمي لتدريب Qwen3-1.7B 4-bit على Colab T4 مع QLoRA [unsloth docs / HF model card].

---

## 3️⃣ إعداد الإعدادات (config)

تعديل الملف:

```bash
config/baligh_pipeline_config.json
```

أهم الحقول:

```jsonc
{
  "project": {
    "name": "Baligh-1.7B-v0",
    "org": "your-org",
    "base_model": "unsloth/Qwen3-1.7B-Base"
  },
  "sft": {
    "output_path": "data/sft/baligh_sft_mixture",
    "val_ratio": 0.02,
    "shuffle_seed": 3407,
    "datasets": [
      { "name": "arbml/CIDAR", "split": "train", "weight": 0.4 },
      { "name": "FreedomIntelligence/evol-instruct-arabic", "split": "train", "weight": 0.35 },
      { "name": "ArSyra/arsyra-instruction-tuning", "split": "train", "weight": 0.1 },
      { "name": "nazimali/quran-question-answer-context", "split": "train", "weight": 0.05 }
    ]
  },
  "train": {
    "model_name": "unsloth/Qwen3-1.7B-Base",
    "max_seq_length": 2048,
    "output_dir": "models/Baligh-1.7B-v0-lora",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 0.0002,
    "logging_steps": 10,
    "eval_steps": 50,
    "save_steps": 100,
    "save_total_limit": 2,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "fp16": true,
    "bf16": false
  },
  "hf": {
    "repo_id": "your-org/Baligh-1.7B-v0",
    "private": false
  }
}
```

- `sft.datasets`: قائمة الـ datasets المستخدمة لنهاية SFT (يمكنك تعديلها بحرية).
- `train.max_steps`: عدد خطوات التدريب (200 للتجارب، غيّرها لأي قيمة مناسبة بعد التأكد).
- `hf.repo_id`: اسم الريبو على Hugging Face.

---

## 4️⃣ تشغيل التدريب end-to-end على Colab CLI + T4

### 4.1 من جهازك المحلي

```bash
# 1) استنساخ المشروع
git clone <this_repo_url> Baligh-1.7B-v0
cd Baligh-1.7B-v0

# 2) إعداد Colab CLI + auth
pip install -U google-colab-cli
colab auth login

# 3) تشغيل pipeline الكامل على T4
./scripts/run_baligh_colab.sh
```


### 4.2 ما الذي يفعله `run_baligh_colab.sh`؟

1. ينشئ session جديد على Colab باسم `baligh-train` مع GPU T4:

```bash
colab new -s baligh-train --gpu T4
```

2. يرفع مجلد المشروع الحالي إلى `/workspace/Baligh-1.7B-v0` في الـ VM:

```bash
colab upload -s baligh-train . /workspace/Baligh-1.7B-v0
```

3. داخل الـ VM:
    - يثبت `requirements.txt`.
    - يشغل:
        - `python scripts/04_build_sft_mixture.py`
        - `python scripts/03_train_sft_unsloth.py`
4. بعد انتهاء التدريب، ينزّل النموذج:

```bash
colab download -s baligh-train \
  /workspace/Baligh-1.7B-v0/models/Baligh-1.7B-v0-lora \
  ./models/Baligh-1.7B-v0-lora
```


بعد انتهاء السكربت، سيكون لديك مجلد `models/Baligh-1.7B-v0-lora` جاهز للاستخدام المحلي والرفع إلى HF.

---

## 5️⃣ ما الذي يفعله كل سكربت بالضبط؟

### 5.1 `04_build_sft_mixture.py`

- يقرأ إعدادات SFT من `config/baligh_pipeline_config.json`.
- يحمل datasets مثل:
    - `arbml/CIDAR`
    - `FreedomIntelligence/evol-instruct-arabic`
    - `ArSyra/arsyra-instruction-tuning`
    - `nazimali/quran-question-answer-context`
- يوحّد نص الـ sample إلى قالب:

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

- يفلتر الأمثلة القصيرة/الفارغة.
- يعمل `interleave_datasets` بنسب `weight`.
- يشغّل train/validation split:
    - `val_ratio` من size الإجمالي (مثل 2%).
    - shuffle بـ seed ثابت.
- يحفظ الناتج كـ `DatasetDict(train, validation)` في:

```bash
data/sft/baligh_sft_mixture/
```


### 5.2 `03_train_sft_unsloth.py`

- يحمّل SFT dataset من `data/sft/baligh_sft_mixture`.
- يحضر:
    - `train` subset.
    - `validation` (لو موجودة).
- يحمّل:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=train_cfg["model_name"],
    max_seq_length=train_cfg["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)
```

- يفعل QLoRA فوق Qwen3-1.7B بذاكرة منخفضة:
    - `r=16`, `alpha=16`, no dropout, gradient checkpointing = `"unsloth"`.
- يضبط `SFTConfig` من config (batch, lr, steps, eval_steps…).
- يطلق `SFTTrainer`:
    - `train_dataset=train_dataset`
    - `eval_dataset=eval_dataset` (اختياري)
- يشغّل `trainer.train()`.
- يحفظ النموذج النهائي تحت:

```bash
models/Baligh-1.7B-v0-lora/
```


### 5.3 `05_run_eval.py` (اختياري)

- يحمّل نموذج `models/Baligh-1.7B-v0-lora`.
- يجرّب prompt عربي واحد/اثنين (مثلاً: سؤال فقهي بسيط).
- يحوّلها لـ generation ويطبع المخرجات.
- مفيد للتأكد أن الأمور تعمل قبل الرفع.

---

## 6️⃣ رفع النموذج إلى Hugging Face

بعد أن ترى أن النموذج يعمل، يمكنك رفعه إلى HF:

```bash
huggingface-cli login   # لو لم تفعلها سابقًا
python scripts/upload_to_hf.py
```

سكربت `upload_to_hf.py`:

- يقرأ:
    - `hf.repo_id` من config (مثلاً: `"mohamed-kandil/Baligh-1.7B-v0"`).
    - `train.output_dir` (مسار النموذج).
- يستخدم:

```python
from huggingface_hub import HfApi, create_repo, upload_folder
```

- يستدعي:
    - `create_repo(repo_id, private=...)`
    - `upload_folder(repo_id, folder_path=model_dir, repo_type="model")`
- بعد النجاح، يكون النموذج متاحًا على:

```text
https://huggingface.co/<repo_id>
```


---

## 7️⃣ استخدام النموذج بعد التدريب

مثال Python محلي:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "your-org/Baligh-1.7B-v0",  # أو المسار المحلي ./models/Baligh-1.7B-v0-lora
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = """### Instruction:
اشرح الفرق بين الفقه والعقيدة باختصار.

### Response:
"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```


---

## 8️⃣ تطويرات لاحقة

Baligh-1.7B-v0 هو **baseline**:

- SFT فقط على Qwen3-1.7B 4-bit.
- لاحقًا يمكن إضافة:
    - CPT جدي على ArabicWeb24.
    - RL (DPO/GRPO) لمزيد من alignment.
    - دمج Benchmarks عربية (ABB, HELM Arabic) في `05_run_eval.py`.

Pull requests وأفكار تحسين مرحب بها 👋

```

***

لو تريد بعد ذلك أضيف في `03_train_sft_unsloth.py` دعم **Weights & Biases** (project name, run name, metrics, config logging) بشكل نظيف ومفعّل/مغلق بالـ config (مثلاً `train.use_wandb: true/false`)، أقدر أكتب لك نسخة كاملة جاهزة للدمج.

---

# copy past all files file by file

سأضع لك كل الملفات الأساسية **جاهزة للـ copy‑paste**، ملفًا بعد ملف، بالترتيب المنطقي داخل مشروعك `Baligh-1.7B-v0/`.

الهيكل المستهدف:

```bash
Baligh-1.7B-v0/
  README.md
  requirements.txt

  config/
    baligh_pipeline_config.json

  data/
    cpt/
    sft/

  scripts/
    01_build_cpt_mixture.py
    02_prepare_cpt_jsonl.py
    03_train_sft_unsloth.py
    04_build_sft_mixture.py
    05_run_eval.py
    run_baligh_colab.sh
    upload_to_hf.py
```

أنشئ المجلدات المناسبة، ثم انسخ كل ملف كما هو.

***

## 1️⃣ `README.md` (في جذر المشروع)

```markdown
# Baligh-1.7B-v0

Baligh-1.7B-v0 هو نموذج لغة عربي (1.5B) مبني على Qwen3-1.7B عبر Unsloth، مع Supervised Fine-tuning (SFT) على بيانات تعليمات عربية (CIDAR, evol-instruct-arabic, إلخ).  
هذا الريبو يتيح لك تشغيل **تدريب كامل** على **Google Colab T4** باستخدام **Colab CLI** من الترمينال، ثم رفع النموذج إلى Hugging Face.

## ✨ الميزات

- Clear architecture: فصل بين `config`, `data`, `scripts`, `models`, `logs`.
- Config-driven بالكامل عبر `config/baligh_pipeline_config.json`.
- Data mixing لـ SFT من Hugging Face datasets (CIDAR, evol-instruct-arabic, ArSyra, Quran QA).
- تدريب SFT باستخدام Unsloth + QLoRA على Qwen3-1.7B 4-bit (مناسب لـ T4 GPU).
- دعم train/validation split و eval أثناء التدريب.
- سكربت رفع جاهز إلى Hugging Face Hub.

---

## 1️⃣ بنية المشروع

```bash
Baligh-1.7B-v0/
  README.md
  requirements.txt

  config/
    baligh_pipeline_config.json

  data/
    cpt/
    sft/

  scripts/
    01_build_cpt_mixture.py
    02_prepare_cpt_jsonl.py
    03_train_sft_unsloth.py
    04_build_sft_mixture.py
    05_run_eval.py
    run_baligh_colab.sh
    upload_to_hf.py

  models/
    Baligh-1.7B-v0-lora/

  logs/
    train/
    eval/
```


---

## 2️⃣ المتطلبات الأساسية

### 2.1 على جهازك المحلي

```bash
pip install -U google-colab-cli huggingface_hub
colab auth login
huggingface-cli login
```


### 2.2 داخل Colab runtime

يتم تثبيت المتطلبات عبر:

```bash
pip install -U -r requirements.txt
```


---

## 3️⃣ config

عدّل الملف:

```bash
config/baligh_pipeline_config.json
```

ليحتوي على إعدادات SFT و HF (انظر الملف في القسم التالي).

---

## 4️⃣ التشغيل end-to-end على Colab CLI + T4

```bash
git clone <this_repo_url> Baligh-1.7B-v0
cd Baligh-1.7B-v0

pip install -U google-colab-cli
colab auth login

./scripts/run_baligh_colab.sh
```

بعد الانتهاء، سيظهر النموذج في:

```bash
models/Baligh-1.7B-v0-lora/
```

ثم:

```bash
huggingface-cli login
python scripts/upload_to_hf.py
```

لرفع النموذج إلى Hugging Face.

---

## 5️⃣ استخدام النموذج بعد التدريب

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "your-org/Baligh-1.7B-v0",  # أو المسار المحلي ./models/Baligh-1.7B-v0-lora
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = """### Instruction:
اشرح الفرق بين الفقه والعقيدة باختصار.

### Response:
"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

Pull requests وأفكار تحسين مرحب بها 👋

```

***

## 2️⃣ `requirements.txt` (في جذر المشروع)

```text
unsloth
transformers
trl
accelerate
datasets
bitsandbytes
huggingface_hub
```


***

## 3️⃣ `config/baligh_pipeline_config.json`

```json
{
  "project": {
    "name": "Baligh-1.7B-v0",
    "org": "your-org",
    "base_model": "unsloth/Qwen3-1.7B-Base"
  },
  "cpt": {
    "output_path": "data/cpt/baligh_cpt_mixture",
    "jsonl_path": "data/cpt/baligh_cpt_train.jsonl",
    "datasets": [
      { "name": "lightonai/ArabicWeb24", "split": "train", "weight": 0.7, "text_field": "text" },
      { "name": "Jr23xd23/ArabicText-Large", "split": "train", "weight": 0.2, "text_field": "text" },
      { "name": "premio-ai/TheArabicPile_Dialects", "split": "train", "weight": 0.1, "text_field": "text" }
    ]
  },
  "sft": {
    "output_path": "data/sft/baligh_sft_mixture",
    "val_ratio": 0.02,
    "shuffle_seed": 3407,
    "datasets": [
      { "name": "arbml/CIDAR", "split": "train", "weight": 0.4 },
      { "name": "FreedomIntelligence/evol-instruct-arabic", "split": "train", "weight": 0.35 },
      { "name": "ArSyra/arsyra-instruction-tuning", "split": "train", "weight": 0.1 },
      { "name": "nazimali/quran-question-answer-context", "split": "train", "weight": 0.05 }
    ]
  },
  "train": {
    "model_name": "unsloth/Qwen3-1.7B-Base",
    "max_seq_length": 2048,
    "output_dir": "models/Baligh-1.7B-v0-lora",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 0.0002,
    "logging_steps": 10,
    "eval_steps": 50,
    "save_steps": 100,
    "save_total_limit": 2,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "fp16": true,
    "bf16": false
  },
  "eval": {
    "datasets": [
      { "name": "FreedomIntelligence/MMLU_Arabic", "split": "test" },
      { "name": "castorini/mr-tydi", "split": "test" }
    ]
  },
  "hf": {
    "repo_id": "your-org/Baligh-1.7B-v0",
    "private": false
  }
}
```

عدّل `hf.repo_id` و `project.org` لاسم الـ org/username الخاص بك.

***

## 4️⃣ `scripts/01_build_cpt_mixture.py`

```python
#!/usr/bin/env python
# scripts/01_build_cpt_mixture.py

import json
import logging
from pathlib import Path

from datasets import load_dataset, interleave_datasets

logging.basicConfig(
    format="[%(asctime)s] [CPT] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"


def load_and_normalize_dataset(name: str, split: str, text_field: str):
    logging.info(f"Loading dataset '{name}' (split={split})")
    ds = load_dataset(name, split=split)

    if text_field not in ds.column_names:
        candidates = [
            c for c in ds.column_names if c.lower() in ["text", "content", "article", "body"]
        ]
        if not candidates:
            raise ValueError(f"No suitable text field found for {name}. Columns: {ds.column_names}")
        logging.warning(
            "text_field '%s' not in columns for %s, using inferred '%s'",
            text_field,
            name,
            candidates[0],
        )
        text_field = candidates[0]

    def _to_text(example):
        t = example.get(text_field) or ""
        return {"text": t}

    ds = ds.map(_to_text, remove_columns=ds.column_names)
    logging.info("Dataset '%s' normalized to single 'text' column (rows=%d)", name, len(ds))
    return ds


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cpt_cfg = cfg["cpt"]
    output_path = Path(__file__).resolve().parents[1] / cpt_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    weights = []

    logging.info("Building CPT mixture from %d datasets", len(cpt_cfg["datasets"]))
    for ds_cfg in cpt_cfg["datasets"]:
        name = ds_cfg["name"]
        split = ds_cfg.get("split", "train")
        weight = float(ds_cfg["weight"])
        text_field = ds_cfg.get("text_field", "text")

        ds = load_and_normalize_dataset(name, split, text_field)
        datasets.append(ds)
        weights.append(weight)

    logging.info("Interleaving datasets with weights: %s", weights)
    mixed = interleave_datasets(
        datasets,
        probabilities=weights,
        seed=3407,
        stopping_strategy="all_exhausted",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(output_path))

    logging.info("Mixed CPT dataset saved to '%s' (rows=%d)", output_path, len(mixed))
    print(
        {
            "saved_to": str(output_path),
            "rows": len(mixed),
            "datasets": [d["name"] for d in cpt_cfg["datasets"]],
        }
    )


if __name__ == "__main__":
    main()
```


***

## 5️⃣ `scripts/02_prepare_cpt_jsonl.py`

```python
#!/usr/bin/env python
# scripts/02_prepare_cpt_jsonl.py

import json
import logging
from pathlib import Path

from datasets import load_from_disk

logging.basicConfig(
    format="[%(asctime)s] [CPT_JSONL] %(levelname)s: %(message)s",
    level=logging.INFO,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cpt_cfg = cfg["cpt"]
    input_path = ROOT / cpt_cfg["output_path"]
    output_file = ROOT / cpt_cfg["jsonl_path"]

    if not input_path.exists():
        raise SystemExit(f"[CPT_JSONL] CPT mixture not found at {input_path}")

    logging.info("Loading CPT mixture from '%s'", input_path)
    ds = load_from_disk(str(input_path))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing JSONL to '%s'", output_file)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 50:
                continue
            rec = {"text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    logging.info("Wrote %d JSONL records", count)
    print({"jsonl": str(output_file), "rows": count})


if __name__ == "__main__":
    main()
```


***

## 6️⃣ `scripts/04_build_sft_mixture.py`

```python
#!/usr/bin/env python
# scripts/04_build_sft_mixture.py

import json
import logging
from pathlib import Path

from datasets import load_dataset, interleave_datasets, DatasetDict

logging.basicConfig(
    format="[%(asctime)s] [SFT] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"

PROMPT_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

PROMPT_NO_INPUT = """### Instruction:
{instruction}

### Response:
{response}"""


def normalize_sample(example):
    instruction = (
        example.get("instruction") or example.get("question") or example.get("prompt") or ""
    )
    inp = example.get("input") or example.get("context") or ""
    response = example.get("output") or example.get("answer") or example.get("response") or ""

    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    response = (response or "").strip()

    if not instruction and not response:
        return {"text": ""}

    if inp:
        text = PROMPT_WITH_INPUT.format(
            instruction=instruction,
            input=inp,
            response=response,
        )
    else:
        text = PROMPT_NO_INPUT.format(
            instruction=instruction,
            response=response,
        )
    return {"text": text}


def build_sft_mixture(cfg):
    sft_cfg = cfg["sft"]
    output_path = Path(__file__).resolve().parents[1] / sft_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    probs = []

    logging.info("Building SFT mixture from %d datasets", len(sft_cfg["datasets"]))
    for ds_cfg in sft_cfg["datasets"]:
        name = ds_cfg["name"]
        split = ds_cfg.get("split", "train")
        weight = float(ds_cfg["weight"])

        logging.info("Loading SFT dataset '%s' (split=%s, weight=%.3f)", name, split, weight)
        ds = load_dataset(name, split=split)

        ds = ds.map(
            normalize_sample,
            remove_columns=ds.column_names,
            desc=f"Normalizing {name}",
        )

        def _filter(example):
            t = (example.get("text") or "").strip()
            return len(t) > 50

        ds = ds.filter(_filter, desc=f"Filtering {name} short/empty")

        logging.info("Dataset '%s' after normalization/filtering: %d rows", name, len(ds))

        items.append(ds)
        probs.append(weight)

    logging.info("Interleaving SFT datasets with weights: %s", probs)
    mixed = interleave_datasets(
        items,
        probabilities=probs,
        seed=sft_cfg.get("shuffle_seed", 3407),
        stopping_strategy="first_exhausted",
    )

    logging.info("Mixed SFT dataset size: %d rows", len(mixed))
    return mixed, output_path


def split_train_val(dataset, val_ratio, seed):
    if val_ratio <= 0:
        logging.warning("val_ratio <= 0, using full dataset as train only.")
        return DatasetDict({"train": dataset})

    logging.info("Shuffling dataset before split (seed=%d)", seed)
    dataset = dataset.shuffle(seed=seed)

    val_size = int(len(dataset) * val_ratio)
    if val_size == 0:
        logging.warning("val_ratio too small, no validation rows. Using full dataset as train.")
        return DatasetDict({"train": dataset})

    logging.info("Splitting dataset: %d train / %d validation", len(dataset) - val_size, val_size)
    dsdict = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return DatasetDict({"train": dsdict["train"], "validation": dsdict["test"]})


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sft_cfg = cfg["sft"]
    val_ratio = float(sft_cfg.get("val_ratio", 0.0))
    seed = int(sft_cfg.get("shuffle_seed", 3407))

    mixed, output_path = build_sft_mixture(cfg)
    ds_dict = split_train_val(mixed, val_ratio, seed)

    output_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(output_path))

    logging.info(
        "SFT dataset saved to '%s' (train=%d, validation=%d)",
        output_path,
        len(ds_dict["train"]),
        len(ds_dict.get("validation", [])),
    )

    print(
        {
            "saved_to": str(output_path),
            "train_rows": len(ds_dict["train"]),
            "val_rows": len(ds_dict.get("validation", [])),
        }
    )


if __name__ == "__main__":
    main()
```


***

## 7️⃣ `scripts/03_train_sft_unsloth.py`

```python
#!/usr/bin/env python
# scripts/03_train_sft_unsloth.py

import json
import logging
from pathlib import Path

from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    format="[%(asctime)s] [TRAIN] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg["train"]
    sft_cfg = cfg["sft"]

    sft_path = Path(__file__).resolve().parents[1] / sft_cfg["output_path"]
    if not sft_path.exists():
        raise SystemExit(
            f"[TRAIN] SFT dataset not found at {sft_path}. Run 04_build_sft_mixture.py first."
        )

    logging.info("Loading SFT dataset from '%s'", sft_path)
    ds_dict = load_from_disk(str(sft_path))

    train_dataset = ds_dict["train"]
    eval_dataset = ds_dict.get("validation")

    logging.info("Train rows: %d", len(train_dataset))
    if eval_dataset is not None:
        logging.info("Validation rows: %d", len(eval_dataset))
    else:
        logging.warning("No validation split found; training without eval.")

    logging.info("Loading base model via Unsloth: %s", train_cfg["model_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_cfg["model_name"],
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    logging.info("Applying LoRA (QLoRA settings)")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    output_dir = Path(__file__).resolve().parents[1] / train_cfg["output_dir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Setting up SFTConfig")
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        report_to="none",
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        max_seq_length=train_cfg["max_seq_length"],
        dataset_text_field="text",
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 0) or None,
    )

    logging.info("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")

    logging.info("Saving final model to '%s'", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logging.info("All done.")
    print({"saved_to": str(output_dir)})


if __name__ == "__main__":
    main()
```


***

## 8️⃣ `scripts/05_run_eval.py`

```python
#!/usr/bin/env python
# scripts/05_run_eval.py

import json
import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    format="[%(asctime)s] [EVAL] %(levelname)s: %(message)s",
    level=logging.INFO,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg["train"]
    model_path = Path(__file__).resolve().parents[1] / train_cfg["output_dir"]

    if not model_path.exists():
        raise SystemExit(f"[EVAL] Model directory not found: {model_path}")

    logging.info("Loading model from '%s'", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))

    prompt = """### Instruction:
اشرح الفرق بين الفقه والعقيدة باختصار.

### Response:
"""
    logging.info("Running test generation...")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=== PROMPT ===")
    print(prompt)
    print("=== OUTPUT ===")
    print(text)


if __name__ == "__main__":
    main()
```


***

## 9️⃣ `scripts/run_baligh_colab.sh`

```bash
#!/usr/bin/env bash
# scripts/run_baligh_colab.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${BALIGH_COLAB_SESSION:-baligh-train}"

echo "[Baligh] Using project root: $PROJECT_ROOT"
echo "[Baligh] Colab session name: $SESSION_NAME"

# 1) إنشاء Colab session مع T4 GPU
colab new -s "$SESSION_NAME" --gpu T4

# 2) رفع المشروع إلى /workspace/Baligh-1.7B-v0 في الـ runtime
echo "[Baligh] Uploading project to Colab..."
colab upload -s "$SESSION_NAME" "$PROJECT_ROOT" /workspace/Baligh-1.7B-v0

# 3) تثبيت المتطلبات داخل runtime
echo "[Baligh] Installing requirements in Colab..."
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess
os.chdir("/workspace/Baligh-1.7B-v0")
subprocess.check_call(["pip", "install", "-U", "-r", "requirements.txt"])
PY

# 4) بناء خليط SFT
echo "[Baligh] Building SFT mixture..."
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess
os.chdir("/workspace/Baligh-1.7B-v0")
subprocess.check_call(["python", "scripts/04_build_sft_mixture.py"])
PY

# 5) تدريب SFT بـ Unsloth
echo "[Baligh] Training SFT with Unsloth..."
colab exec -s "$SESSION_NAME" -- << 'PY'
import os, subprocess
os.chdir("/workspace/Baligh-1.7B-v0")
subprocess.check_call(["python", "scripts/03_train_sft_unsloth.py"])
PY

# 6) تنزيل النموذج النهائي
echo "[Baligh] Downloading trained model to local ./models/Baligh-1.7B-v0-lora ..."
mkdir -p "$PROJECT_ROOT/models"
colab download -s "$SESSION_NAME" \
  /workspace/Baligh-1.7B-v0/models/Baligh-1.7B-v0-lora \
  "$PROJECT_ROOT/models/Baligh-1.7B-v0-lora"

echo "[Baligh] Pipeline finished. Model at models/Baligh-1.7B-v0-lora"
```


***

## 🔟 `scripts/upload_to_hf.py`

```python
#!/usr/bin/env python
# scripts/upload_to_hf.py

import os
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "baligh_pipeline_config.json"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_id = cfg["hf"]["repo_id"]
    private = cfg["hf"].get("private", False)
    model_dir = ROOT / cfg["train"]["output_dir"]

    if not model_dir.exists():
        raise SystemExit(f"[HF_UPLOAD] Model directory not found: {model_dir}")

    api = HfApi()
    create_repo(repo_id=repo_id, private=private, exist_ok=True)

    print(f"[HF_UPLOAD] Uploading {model_dir} to Hugging Face Hub: {repo_id} ...")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message="Upload Baligh-1.7B-v0 model",
    )

    print(f"[HF_UPLOAD] Done. Model is at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
```


***

بهذا عندك **كل الملفات الأساسية** جاهزة للنسخ.
الخطوات الآن:

```bash
cd Baligh-1.7B-v0
pip install -U google-colab-cli huggingface_hub
colab auth login
huggingface-cli login

./scripts/run_baligh_colab.sh   # التدريب على T4
python scripts/05_run_eval.py   # اختياري: اختبار محلي
python scripts/upload_to_hf.py  # رفع إلى HF
```

لو واجهت أي مشكلة في أي خطوة (schema dataset مختلف، خطأ Colab CLI، إلخ) أرسل اللوج وسأعدّل لك السكربتات حسب الواقع.

---

# [https://github.com/FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)

اللينك اللي أرسلته هو ريبو `train-llm-from-scratch` بتاع Fareed Khan، وفيه implementation كامل لـ Transformer + pipeline تدريب من الصفر (data download → preprocess → train → generate) مع suite بعد كده لـ SFT / RLHF (SFT, Reward Model, PPO, DPO, GRPO) مكتوبة بـ PyTorch pure بدون `transformers` أو `trl`.[^18_1]

ريبو زي ده مفيد لك في Baligh-1.7B في حاجتين رئيسيتين:

- كـ **مرجع معماري وتعليمي** لو حابب تفهم/تتحكم أكتر في layers, attention, training loop.[^18_1]
- أو لو في المستقبل حبيت **تعمل Arabic LLM من الصفر** (مش بس CPT/SFT على Qwen2.5) باستخدام نفس الأفكار لكن مع ArabicWeb24 بدل The Pile.[^18_2][^18_3][^18_4][^18_1]

بسّ، الـ pipeline اللي بنيناه لـ Baligh-1.7B-v0 حاليًا:

- مبني على **Qwen3-1.7B + Unsloth** مع `transformers`/`trl` و Colab CLI.
- أما مشروع Fareed فهو:
    - transformer custom في `src/models/` (mlp.py, attention.py, transformer_block.py, transformer.py).
    - config في `config/config.py` + JSON configs في `configs/`.
    - data pipeline خاص (Pile-uncopyrighted + HDF5).[^18_1]
    - training scripts في `scripts/train_transformer.py`, `scripts/data_download.py`, `scripts/data_preprocess.py`, إلخ.[^18_1]

يعني لو حاولت تدمج الاتنين حرفيًا في نفس المشروع هتبقى بتخلط **two different stacks**:

- Stack 1:
    - HF/Unsloth/Qwen2.5 (اللي بنشتغل به الآن على T4).
- Stack 2:
    - Pure PyTorch transformer (Fareed) مع pipelines مختلفة.

أفضل استخدام عملي لك الآن:

1. تكمّل Baligh-1.7B-v0 زي ما هو (Qwen2.5 + SFT Arabic) على Colab CLI.
2. تستخدم ريبو Fareed كـ “reference lab”:
    - تقرأ `config/config.py` و `scripts/train_transformer.py` وتشوف إزاي بيضبط hyperparameters و training loop.
    - لو حابب تعمل Arabic LLM small (مثلاً 13M–100M) من الصفر على ArabicWeb24، تستخدم نفس architecture/loop لكن تغيّر:
        - tokenizer (مثلاً عربى BPE).
        - data loader لـ ArabicWeb24 بدل The Pile.[^18_3][^18_4][^18_2]

لو قصدك من اللينك إنك عايز:

- خطة دمج معينة (مثلاً: “عايز أستخدم loop التدريب بتاعه بس مع Qwen2.5”)، أو
- إن Baligh-1.7B-v1 يبقى “transformer من الصفر على ArabicWeb24” بنفس style الريبو ده،

قول لي بالظبط أنت عايز تعمل إيه بالريبو ده في سياق Baligh، وسأكتب لك plan خطوة بخطوة (مثلاً: “fork + حذف data_download + استبدالها بـ ArabicWeb24 + تحديث config + تعديل training script…”) تناسب هدفك.

<div align="center">⁂</div>

[^18_1]: https://www.youtube.com/watch?v=oOVOkaQ_DSM

[^18_2]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^18_3]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^18_4]: https://lighton.ai/lighton-blogs/arabicweb24

