# Rung 2e - Full RL with GRPO (fine-tuning becomes a system you operate)

**First principles.** Train against a reward signal in a real environment. The model generates
rollouts (candidate SQL), the environment scores them (execute against SQLite, reward = the right
rows), and the trainer updates weights to favor higher-reward rollouts, all kept in sync. This is
the most powerful and by far the most infrastructure-heavy step.

**What you do.** Run GRPO from the 2d (or 2b) checkpoint with `common.sql_executor.reward` as the
reward function: 1.0 for the gold rows, 0.1 for valid-but-wrong (a smoother early signal), 0.0 for
broken SQL. vLLM generates the rollouts. Watch mean reward climb over steps.

**What to watch.** EX pushes past the SFT/DPO ceiling on the hardest queries. More importantly, you
feel the operational shift: you are no longer submitting a training job, you are running a
generate-score-update loop with a model server, an environment, and a trainer that all have to stay
in sync.

**Customer lesson.** This is exactly what Cursor and Kimi do, and it is where a customer can no
longer use a managed fine-tuning endpoint at all. The artifact *and* the training system are
bespoke.

**Artifact.** A checkpoint. Published to Releases, not committed.

Run:
```bash
bash rung2e-grpo/run.sh
```
