# Rung 1 - Vendor fine-tune (the rented-customization trap)

**Concepts (from first principles).**
- *Supervised fine-tuning (SFT)*: you hand the vendor labeled examples (here: question+schema ->
  correct SQL) and it adjusts the model's weights to imitate them. Unlike prompting (rung 0), the
  weights actually change. Unlike rung 2, the change happens *inside the vendor* and you never
  receive the resulting weights.
- *Base vs tuned*: the "base" model is the untouched hosted model; the "tuned" model is your
  fine-tuned version. The lift between them, measured the same way, is what SFT bought.
- *Managed / hosted tuning*: the whole job runs on the vendor's platform -- you upload data, they
  train and host the result. Convenient, but portable nowhere.

**First principles.** The vendor adjusts weights for you, but you never receive them. You are
renting a *customized* rental.

**What you do.** We fine-tune **Gemini Flash in Google Vertex AI** (OpenAI's fine-tuning platform is
closed -- see below). `finetune_vertex.py` builds a tuning JSONL from Spider train, uploads it to a
GCS bucket, runs a supervised tuning job, then evaluates two rows through the frozen eval, zero-shot:
`base-gemini` (untuned) and `vendor-ft-gemini` (tuned). Because the same Gemini Flash is the untuned
frontier row in rung 0, this is a clean within-model, same-vendor comparison.

**What to watch.** The `base-gemini` -> `vendor-ft-gemini` lift: how much a modest SFT moves a hosted
model on this task. Then the constraint that matters more than the number: even when it succeeds, the
tuned weights live in Vertex. You cannot download them; the customization is portable nowhere.

**Customer lesson.** The moment your differentiation is real, you do not want it trapped inside
someone else's platform. This rung is the argument for open weights, and the reason the journey
continues to rung 2.

**The OpenAI door is closed (the sharper version of the lesson).** As of mid-2026 OpenAI wound its
fine-tuning platform down: job creation returns `403 training_not_available`. If `OPENAI_API_KEY` is
set, `run.sh` runs `finetune_openai.py` as an aside -- it records the `base-mini` baseline, then
prints that refusal and stops. A vendor revoking your ability to customize, mid-flight, is the
rented-customization trap in its purest form: you did not just fail to get the weights, you lost the
option entirely. That is precisely why the next rung is to own them. Other managed vendors still
offer hosted tuning (AWS Bedrock, Mistral, Cohere, Azure OpenAI) with the same "no weights"
property; we use Vertex/Gemini here because it doubles as the rung-0 frontier model.

**Artifact.** Tuned weights that live inside the vendor, portable nowhere.

Run:
```bash
# GCP application-default creds: gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=... VERTEX_LOCATION=us-central1 T2S_GCS_BUCKET=your-bucket
export OPENAI_API_KEY=...   # optional: also runs the closed-door OpenAI aside
bash rung1-vendor-ft/run.sh
```
