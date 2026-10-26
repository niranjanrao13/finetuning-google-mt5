import os
import time, random, torch
import pandas as pd
import sacrebleu
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, evaluate
from config import N_EVAL, MAX_WORKERS

from config import (
    trained_model_checkpoint, max_input_length, max_target_length,
    source_lang, target_lang, prefix,
)

# ------------------------------------------------------------------ constants

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

PROMPT = "Translate the following English text to Arabic. Output only the translation.\n\n{text}"

# label -> (provider, base_url, model_id, api_key). Entries whose key is still a placeholder are skipped.
LLMS = {
    # "Claude Haiku 4.5":  ("anthropic", None,                      "claude-haiku-4-5",  ANTHROPIC_API_KEY),
    # "Claude Sonnet 5":   ("anthropic", None,                      "claude-sonnet-5-5", ANTHROPIC_API_KEY),
    # "Grok (xAI)":        ("openai",    "https://api.x.ai/v1",     "grok-4.3",          XAI_API_KEY),
    "GPT-OSS-120B":      ("openai",    None, "openai/gpt-oss-120b", GROQ_API_KEY),
}

# ------------------------------------------------------------------ data
def load_test_data():
    ds = load_dataset('Helsinki-NLP/opus-100', 'ar-en')
    test_subset = ds["test"].select(range(N_EVAL))
    srcs = [ex[source_lang] for ex in test_subset["translation"]]
    refs = [ex[target_lang] for ex in test_subset["translation"]]
    return srcs, refs

# ------------------------------------------------------------------ local model
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(trained_model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer, device

def translate_local(model, tokenizer, device, texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(
            [prefix + t for t in texts[i:i + bs]],
            return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_length,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_target_length, num_beams=4)
        out += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return out

# ------------------------------------------------------------------ API models
def make_translator(provider, base_url, model_id, key):
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
        def call(text):
            r = client.messages.create(
                model=model_id, max_tokens=512,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
            )
            return "".join(b.text for b in r.content if b.type == "text").strip()
    else:
        client = Groq(api_key=key)
        def call(text):
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
            )
            return (r.choices[0].message.content or "").strip()
    return call

_shown_errors = set()
def with_retry(call, text, tries=5):
    for k in range(tries):
        try:
            return call(text)
        except Exception as e:
            if k == tries - 1:
                msg = repr(e)[:200]
                if msg not in _shown_errors:
                    _shown_errors.add(msg); print("  request failed:", msg)
                return ""
            time.sleep(2 ** k + random.random())

def translate_api(call, texts):
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        return list(pool.map(lambda t: with_retry(call, t), texts))

# ------------------------------------------------------------------ scoring
def score(hyps, refs, metric):
    bleu = metric.compute(predictions=hyps, references=[[r] for r in refs])["score"]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    return round(bleu, 2), round(chrf, 2)

# ------------------------------------------------------------------ main
def main():
    srcs, refs = load_test_data()
    model, tokenizer, device = load_local_model()
    metric = evaluate.load("sacrebleu")

    all_hyps, rows = {}, []

    print(f"Translating {N_EVAL} sentences with my fine-tuned mT5...")
    all_hyps["My mT5-small (fine-tuned)"] = translate_local(model, tokenizer, device, srcs)

    for label, (provider, base_url, model_id, key) in LLMS.items():
        if key.startswith("YOUR_"):
            print(f"Skipping {label}: API key placeholder not filled in")
            continue
        print(f"Translating with {label} ({model_id})...")
        all_hyps[label] = translate_api(make_translator(provider, base_url, model_id, key), srcs)

    for label, hyps in all_hyps.items():
        bleu, chrf = score(hyps, refs, metric)
        rows.append({"model": label, "BLEU": bleu, "chrF++": chrf,
                     "empty outputs": sum(h == "" for h in hyps)})

    results = pd.DataFrame(rows).sort_values("BLEU", ascending=False).reset_index(drop=True)
    print(results.to_string(index=False))

    for i in range(3):
        print(f"\nEN : {srcs[i]}\nREF: {refs[i]}")
        for label, hyps in all_hyps.items():
            print(f"{label[:26]:<26}: {hyps[i]}")


if __name__ == "__main__":
    main()