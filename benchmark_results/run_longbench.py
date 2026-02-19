#!/usr/bin/env python3
"""Run LongBench2 benchmark with timing metrics."""
import os, json, time, sys, re
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

# Config
URL = "http://10.2.10.55:8080/v1"
API_KEY = "token-abc123"
MODEL_KEY = "Qwen2.5-72B-Instruct"
SAVE_DIR = "/workspace/results"
MAX_EXAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 503  # pass N to limit

os.chdir("/workspace/LongBench")
model_map = json.loads(open("config/model2path.json").read())
maxlen_map = json.loads(open("config/model2maxlen.json").read())
template = open("prompts/0shot.txt").read()

model_path = model_map[MODEL_KEY]
max_len = maxlen_map[MODEL_KEY]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
client = OpenAI(base_url=URL, api_key=API_KEY)

dataset = load_dataset("THUDM/LongBench-v2", split="train")
data_all = [dict(item) for item in dataset]

# Resume support
os.makedirs(SAVE_DIR, exist_ok=True)
out_file = os.path.join(SAVE_DIR, f"{MODEL_KEY}.jsonl")
has_data = set()
if os.path.exists(out_file):
    with open(out_file) as f:
        for line in f:
            has_data.add(json.loads(line)["_id"])
    print(f"Resuming: {len(has_data)} already completed")

fout = open(out_file, "a")
data = [item for item in data_all if item["_id"] not in has_data]
data = data[:MAX_EXAMPLES]

def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None

print(f"Model: {model_path}")
print(f"Max tokens: {max_len}")
print(f"Pending: {len(data)} / {len(data_all)} total")
print(f"Output: {out_file}")
print()

timings = []
correct = 0
total = 0

for i, item in enumerate(tqdm(data, desc="LongBench2")):
    context = item["context"]
    prompt = (template
        .replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip()))

    # Truncate
    input_ids = tokenizer.encode(prompt)
    orig_tokens = len(input_ids)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    final_tokens = len(input_ids)

    start = time.time()
    tries = 0
    response = ""
    while tries < 3:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=128,
            )
            response = completion.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"\n  Error on Q{i+1} (try {tries}): {e}")
            time.sleep(2)

    elapsed = time.time() - start
    timings.append({"tokens": final_tokens, "time": elapsed, "length": item["length"]})

    pred = extract_answer(response)
    judge = pred == item["answer"]
    if judge:
        correct += 1
    total += 1

    item["response"] = response
    item["pred"] = pred
    item["judge"] = judge
    item["context"] = context[:500]  # truncate for storage
    item["_tokens"] = final_tokens
    item["_orig_tokens"] = orig_tokens
    item["_time_s"] = round(elapsed, 2)
    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    fout.flush()

    if (i + 1) % 20 == 0:
        avg_time = sum(t["time"] for t in timings) / len(timings)
        print(f"\n  Progress: {i+1}/{len(data)} | Acc: {correct}/{total} ({100*correct/total:.1f}%) | Avg time: {avg_time:.1f}s")

fout.close()

# Summary
print("\n" + "=" * 60)
print("LONGBENCH2 RESULTS")
print("=" * 60)
print(f"Total: {total} questions")
print(f"Correct: {correct} ({100*correct/total:.1f}%)")
avg_time = sum(t["time"] for t in timings) / len(timings)
print(f"Avg time per question: {avg_time:.1f}s")
print(f"Total time: {sum(t['time'] for t in timings):.0f}s")

# By length bucket
for bucket_name, bucket_fn in [
    ("Short (0-32k)", lambda x: "k" in x and int(x.replace("k","").replace("-","").split()[0] if x[0].isdigit() else "0") < 32),
    ("Medium (32k-128k)", lambda x: "k" in x and 32 <= int(x.replace("k","").replace("-","").split()[0] if x[0].isdigit() else "0") < 128),
    ("Long (128k+)", lambda x: "k" in x and int(x.replace("k","").replace("-","").split()[0] if x[0].isdigit() else "0") >= 128),
]:
    pass  # length classification is complex, skip for now

# Timing stats
token_buckets = [(0, 5000), (5000, 15000), (15000, 30000)]
for lo, hi in token_buckets:
    bucket_times = [t for t in timings if lo <= t["tokens"] < hi]
    if bucket_times:
        avg = sum(t["time"] for t in bucket_times) / len(bucket_times)
        avg_tok = sum(t["tokens"] for t in bucket_times) / len(bucket_times)
        print(f"  {lo//1000}k-{hi//1000}k tokens: {len(bucket_times)} questions, avg={avg:.1f}s, avg_tokens={avg_tok:.0f}")
