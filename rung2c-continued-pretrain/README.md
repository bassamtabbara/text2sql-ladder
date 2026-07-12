# Rung 2c - Continued pretraining (optional, the knowledge-altering step)

**First principles.** Before any task tuning, keep *pre*-training the base on a large corpus of
in-domain text to shift its underlying distribution. This is the heaviest, most knowledge-altering
step. It is only worth it when you have hundreds of millions of tokens of in-domain text and the
domain is genuinely far from the base model's world (niche science, a low-resource language, an
unusual code style).

**What you do.** Continue-pretrain on raw schema text plus gold SQL (no chat template, no question),
then re-run the 2b SFT on top of the shifted base and evaluate. Run it mainly to feel that it is
heavy and usually skippable.

**What to watch.** For Spider this is likely marginal, because SQL is not far from Qwen's world. The
lesson is the cost/benefit judgment, not a big number.

**Customer lesson.** This is the Cursor move: continued-pretrain on a code-heavy mix, then
long-context extension, then SFT, and only then RL. Most customers correctly skip this rung, and
being able to say *why* is the point.

**Artifact.** A new base checkpoint. Published to Releases, not committed.

Run:
```bash
bash rung2c-continued-pretrain/run.sh
```
