"""Rung 1 (real SFT): supervised fine-tuning of Gemini Flash in Google Vertex AI.

OpenAI's fine-tuning door is closed (see finetune_openai.py, which now just records the 403), so the
working "what does vendor fine-tuning buy" number comes from Vertex. The rung-1 lesson still holds:
the tuned weights live inside Vertex, you don't get to download them.

Gemini Flash also appears untuned in rung 0, so the comparison here is clean and within-model:
  1, base-gemini      -- untuned model, zero-shot
  1, vendor-ft-gemini -- the SFT'd model, zero-shot
Both go through the frozen run_eval/record_result, scored on the same dev split as every other rung.

Uses the google-genai SDK (the supported one; the older vertexai.generative_models SDK is retired
and only hits regional endpoints, which 404 for models served via the `global` endpoint).

Requires GCP: application-default credentials (or a service account via GOOGLE_APPLICATION_CREDENTIALS),
GOOGLE_CLOUD_PROJECT, and a writable GCS bucket (T2S_GCS_BUCKET). Inference for the base model runs
against VERTEX_INFER_LOCATION (default `global`); tuning and the tuned model run in VERTEX_LOCATION
(default `us-central1`).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from common.data import load_dev_subset, load_split
from common.eval import record_result, run_eval
from common.model_client import SYSTEM_PROMPT, _user_turn


def build_vertex_jsonl(n: int, path: str) -> str:
    """Vertex SFT format: optional systemInstruction + a user->model contents turn per line."""
    train = load_split("train")[:n]
    with open(path, "w") as f:
        for ex in train:
            rec = {
                "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [
                    {"role": "user", "parts": [{"text": _user_turn(ex)}]},
                    {"role": "model", "parts": [{"text": ex.gold_sql}]},
                ],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {n} tuning examples to {path}")
    return path


def upload_to_gcs(local_path: str, bucket: str, blob_name: str) -> str:
    from google.cloud import storage

    client = storage.Client()
    client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)
    uri = f"gs://{bucket}/{blob_name}"
    print(f"uploaded -> {uri}")
    return uri


class VertexGenAIClient:
    """Adapter so the frozen run_eval can drive a Gemini model (base or tuned) via google-genai.

    Implements .complete(messages) by translating OpenAI-style chat messages into Gemini's
    system_instruction + contents, returning raw text (run_eval extracts the SQL).
    """

    def __init__(self, genai_client, model_name: str, temperature: float = 0.0,
                 max_tokens: int = 512):
        self._client = genai_client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], **_overrides) -> str:
        from google.genai import types

        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        contents = [
            {"role": ("user" if m["role"] == "user" else "model"),
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return resp.text or ""


_RUNNING = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED", "JOB_STATE_UPDATING"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VERTEX_MODEL", "gemini-3.5-flash"),
                    help="Vertex source model (must support supervised tuning)")
    ap.add_argument("--n", type=int, default=2000, help="training examples")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--jsonl", default="rung1-vendor-ft/vertex_train.jsonl")
    ap.add_argument("--tuned-endpoint", default=None,
                    help="skip tuning and evaluate an already-tuned endpoint/model name")
    args = ap.parse_args()

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    tune_location = os.environ.get("VERTEX_LOCATION", "us-central1")
    infer_location = os.environ.get("VERTEX_INFER_LOCATION", "global")
    bucket = os.environ["T2S_GCS_BUCKET"]

    from google import genai
    from google.genai import types

    dev = load_dev_subset()

    # 1. Within-model baseline: untuned model, zero-shot. Base models serve from the global endpoint.
    infer_client = genai.Client(vertexai=True, project=project, location=infer_location)
    base_metrics = run_eval(VertexGenAIClient(infer_client, args.model), dev)
    record_result("1", "base-gemini", base_metrics, args.model, notes="zero-shot, no fine-tune")

    # 2. Supervised fine-tune in a region (unless an already-tuned endpoint was passed).
    tuned_endpoint = args.tuned_endpoint
    if tuned_endpoint is None:
        gcs_uri = upload_to_gcs(build_vertex_jsonl(args.n, args.jsonl), bucket,
                                "text2sql-ladder/vertex_train.jsonl")
        tune_client = genai.Client(vertexai=True, project=project, location=tune_location)
        print(f"launching SFT on {args.model} in {tune_location} ({args.epochs} epochs)...")
        try:
            job = tune_client.tunings.tune(
                base_model=args.model,
                training_dataset=types.TuningDataset(gcs_uri=gcs_uri),
                config=types.CreateTuningJobConfig(
                    epoch_count=args.epochs, tuned_model_display_name="text2sql-gemini-sft"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"could not start tuning for {args.model}: {exc}")
            print("If this is a 'model not tunable' error, set VERTEX_MODEL to a tunable id "
                  "(e.g. gemini-2.5-flash) and re-run. base-gemini is already recorded.")
            return
        while str(job.state).split(".")[-1] in _RUNNING:
            time.sleep(60)
            job = tune_client.tunings.get(name=job.name)
            print(f"  tuning: {job.state}")
        if not str(job.state).split(".")[-1].endswith("SUCCEEDED"):
            raise SystemExit(f"tuning did not succeed: {job.state}")
        tuned_endpoint = getattr(job.tuned_model, "endpoint", None) \
            or getattr(job.tuned_model, "model", None)
        print(f"tuned endpoint: {tuned_endpoint}  (note: weights stay in Vertex)")

    # 3. Evaluate the tuned model, zero-shot, in its region -- same inference as the baseline.
    tuned_client = genai.Client(vertexai=True, project=project, location=tune_location)
    tuned_metrics = run_eval(VertexGenAIClient(tuned_client, tuned_endpoint), dev)
    record_result("1", "vendor-ft-gemini", tuned_metrics, args.model,
                  notes=f"SFT on Vertex, n={args.n}, epochs={args.epochs}, weights not portable")


if __name__ == "__main__":
    main()
