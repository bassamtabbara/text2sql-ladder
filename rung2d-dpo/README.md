# Rung 2d - Preference tuning with DPO (toward "better", not just "correct")

**First principles.** After SFT, shape the model toward preferred behavior by learning from
preference *pairs* (chosen vs rejected) directly, with no separate reward model. DPO is the
lightweight, now-common version of alignment.

**What you do.** Sample several candidate queries per training question from the 2b model, then let
the executor label them: a query returning the gold rows is chosen (prefer the shortest correct
one), a wrong one is rejected. Train with DPO on those pairs, starting from the 2b checkpoint. The
preference signal is manufactured for free by the same executor that scores the eval.

**What to watch.** EX ticks up, and more visibly a class of repeated mistakes fades (for example
consistently getting a join direction wrong). You closed the loop: eval metric becomes training
signal.

**Customer lesson.** This is the common, affordable alignment step most teams reach before
committing to full RL.

**Artifact.** A checkpoint. Published to Releases, not committed.

Run:
```bash
bash rung2d-dpo/run.sh
```
