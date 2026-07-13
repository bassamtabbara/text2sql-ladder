# Rung 1 - Vendor fine-tune (the rented-customization trap)

**First principles.** The vendor adjusts weights for you, but you never receive them. You are
renting a *customized* rental.

**What you do.** Build a small JSONL from Spider train (a few hundred examples), submit a fine-tune
job to OpenAI, wait, then evaluate the resulting `ft:...` model with the same frozen eval. Kept
deliberately small; this rung is about the constraints, not the score.

**What to watch.** A modest EX bump over rung 0. Then the three things that matter more than the
number: you cannot download the weights, the model runs only on the vendor's API, and (as of 2026)
OpenAI is winding new fine-tuning down, with new orgs locked out after May 2026.

**Customer lesson.** The moment your differentiation is real, you do not want it trapped inside
someone else's API. This rung is the argument for open weights, and the reason the journey continues
to rung 2.

**The door may already be closed (and that IS the lesson).** As of mid-2026 OpenAI has wound its
fine-tuning platform down: job creation returns `403 training_not_available` for orgs that can no
longer use it. When that happens, `run.sh` still records the `base-mini` baseline first, then prints
the refusal and stops. A blocked fine-tune is a *stronger* rung-1 result than a successful one: the
customization you rented can be revoked from under you, which is precisely why the next rung is to
own the weights.

If you want an actual managed-fine-tune lift number anyway, other vendors still offer hosted tuning
(same "you don't get the weights" property): **AWS Bedrock** (Nova / Llama / Cohere), **Google
Vertex AI** (Gemini Flash), **Mistral La Plateforme**, **Cohere**, or **Azure OpenAI** (separate
governance from openai.com). Point a small adapter of this script at whichever you have access to.

**Artifact.** Customized weights you can't hold, portable nowhere.

Run:
```bash
export OPENAI_API_KEY=...
bash rung1-vendor-ft/run.sh
```
