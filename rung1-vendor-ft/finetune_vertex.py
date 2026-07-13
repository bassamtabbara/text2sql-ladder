"""Rung 1 (real SFT): supervised fine-tuning of Gemini Flash in Google Vertex AI.

OpenAI's fine-tuning door is closed (see finetune_openai.py, which now just records the 403), so the
working "what does vendor fine-tuning buy" number comes from Vertex. The rung-1 lesson still holds:
the tuned weights live inside Vertex, you don't get to download them.

Gemini Flash also appears untuned in rung 0, so the comparison here is clean and within-model:
  1, base-gemini    -- untuned gemini-3.5-flash, zero-shot
  1, vendor-ft-gemini -- the SFT'd model, zero-shot
Both go through the frozen run_eval/record_result, scored on the same dev split as every other rung.

Requires GCP: application-default credentials (or a service account), GOOGLE_CLOUD_PROJECT,
VERTEX_LOCATION, and a writable GCS bucket (T2S_GCS_BUCKET) for the training JSONL.
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


class VertexChatClient:
    """Adapter so the frozen run_eval can drive a Vertex Gemini model (base or tuned).

    Implements .complete(messages) by translating OpenAI-style chat messages into Gemini's
    system_instruction + contents, and returning the raw text (run_eval extracts the SQL).
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], **_overrides) -> str:
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        contents = [
            {"role": ("user" if m["role"] == "user" else "model"),
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        model = GenerativeModel(self.model_name, system_instruction=system) if system \
            else GenerativeModel(self.model_name)
        resp = model.generate_content(
            contents,
            generation_config=GenerationConfig(
                temperature=self.temperature, max_output_tokens=self.max_tokens),
        )
        return resp.text or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VERTEX_MODEL", "gemini-3.5-flash"),
                    help="Vertex source model (must support supervised tuning)")
    ap.add_argument("--n", type=int, default=2000, help="training examples")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--jsonl", default="rung1-vendor-ft/vertex_train.jsonl")
    ap.add_argument("--tuned-endpoint", default=None,
                    help="skip tuning and evaluate an already-tuned endpoint name")
    args = ap.parse_args()

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    bucket = os.environ["T2S_GCS_BUCKET"]

    import vertexai
    from vertexai.tuning import sft

    vertexai.init(project=project, location=location)
    dev = load_dev_subset()

    # 1. Within-model baseline: untuned model, zero-shot. This is the fair "before".
    base_metrics = run_eval(VertexChatClient(args.model), dev)
    record_result("1", "base-gemini", base_metrics, args.model, notes="zero-shot, no fine-tune")

    # 2. Supervised fine-tune (unless an already-tuned endpoint was passed).
    tuned_endpoint = args.tuned_endpoint
    if tuned_endpoint is None:
        gcs_uri = upload_to_gcs(build_vertex_jsonl(args.n, args.jsonl), bucket,
                                "text2sql-ladder/vertex_train.jsonl")
        print(f"launching SFT on {args.model} ({args.epochs} epochs)...")
        job = sft.train(
            source_model=args.model,
            train_dataset=gcs_uri,
            epochs=args.epochs,
            tuned_model_display_name="text2sql-gemini-sft",
        )
        while not job.has_ended:
            time.sleep(60)
            job.refresh()
            print(f"  tuning state: {job.state}")
        if not job.has_succeeded:
            raise SystemExit(f"tuning job did not succeed: {job.state}")
        tuned_endpoint = job.tuned_model_endpoint_name
        print(f"tuned model endpoint: {tuned_endpoint}  (note: weights stay in Vertex)")

    # 3. Evaluate the tuned model, zero-shot -- same inference as the baseline, so the delta is FT.
    tuned_metrics = run_eval(VertexChatClient(tuned_endpoint), dev)
    record_result("1", "vendor-ft-gemini", tuned_metrics, args.model,
                  notes=f"SFT on Vertex, n={args.n}, epochs={args.epochs}, weights not portable")


if __name__ == "__main__":
    main()
