# Security & Infrastructure Review — Baligh (Aug 2026)

### Context

Baligh (D:\AI\Projects\LLM\Baligh) — Arabic LLM fine-tuning (Qwen3-1.7B, QLoRA/Unsloth) with GitHub Actions CI, Docker, Colab notebooks, Modal serverless. Read-only security review requested covering secrets, CI supply chain, Docker, Modal, deps, notebooks, Makefile, .gitignore/.env.example.

### Explanation

VERDICT: No leaked secrets (working tree AND full git history clean; notebooks have zero stored outputs; .env.example uses placeholders only). Top risks are supply-chain and exposure-class: (1) unsloth installed from git HEAD (requirements/training.txt:14, src/modal/app.py:47) executes mutable third-party code at install time inside environments that hold HF/W&B secrets (self-hosted runners, Colab, Modal images); (2) Modal web endpoints (src/modal/serve.py:25,72) are public GET endpoints with no authentication — cost-abuse and unauthenticated model access once deployed; (3) no .dockerignore + COPY . . in all three Dockerfiles bakes .env/data/wandb/git history into layers when building from repo root; (4) modal/train.py uses os.system() with f-string interpolation of CLI params (lines 48,53,73-93,125-133) and clones unpinned main branch before running it with secrets in env; (5) all workflows lack permissions: blocks and use tag-pinned (not SHA-pinned) actions; workflow_dispatch inputs are interpolated directly into run: blocks (sft-training.yml:42, release.yml:39-68) — script injection class, gated behind write access; (6) all requirements are open-ended >= floors with no lockfile; torch>=2.5.1 floor admits versions vulnerable to CVE-2025-32434 (weights_only bypass, fixed 2.6.0) though no torch.load-on-untrusted-input exists in repo; (7) containers run as root, base images pinned by mutable tags not digests; (8) trust_remote_code=True default (src/baligh/config.py:66) enables hub-repo code execution on every model/tokenizer load. Done well: getpass-based token entry in notebooks, empty notebook outputs, no pull_request_target, no curl|bash, no unsafe deserialization (safetensors safe_serialization=True), Modal Secret.from_name usage, compose passes secrets as runtime env not build args, .gitignore covers wandb/htmlcov/checkpoints/weights.

### Rationale (Why this?)

Severity calibrated assuming repo may become public (Colab links in colab_cli/run.py point at public GitHub blob URLs). Highest-leverage fixes are dependency pinning and Modal auth — both cheap and eliminate the two realistic exfiltration/abuse paths. Revisit after any change to runner topology (e.g., if workflow_dispatch gains fork access) or if repo goes public before remediation.

### Next Steps

Builder implements prioritized remediation list in the review report: pin unsloth to a commit SHA or PyPI version; add auth to Modal endpoints (require_header or reverse-proxy JWT); add .dockerignore; replace os.system with subprocess lists; add permissions: blocks + SHA-pin actions; generate lockfiles (uv pip compile / pip-tools); add USER to Dockerfiles; pin HF revisions; consider branch protection requiring SHA-pinned actions via push rulesets.

---
