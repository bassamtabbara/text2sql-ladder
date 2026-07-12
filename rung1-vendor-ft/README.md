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

**Artifact.** Customized weights you can't hold, portable nowhere.

Run:
```bash
export OPENAI_API_KEY=...
bash rung1-vendor-ft/run.sh
```
