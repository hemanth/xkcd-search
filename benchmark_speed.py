"""Benchmark script to measure TypeSafe AI speed, latency, and throughput on XKCD search."""

import asyncio
import json
import os
import sys
import time
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, TypeSafeClient
from dotenv import load_dotenv

load_dotenv()

BENCHMARK_QUERIES = [
    ("Shuttle Skeleton (#2630)", "the space shuttle is actually a mammal with bones"),
    ("Motion Blur (#2628)", "vibrating rapidly to create blur in fast shutter speed photos"),
    ("d65536 (#2626)", "rolling a die with sixty-five thousand five hundred thirty-six sides"),
    ("Or Whatever (#2629)", "sears tower was the world's tallest tower or building in the 90s"),
    ("Types of Scopes (#2627)", "regular and electron types of scopes like telescope and kaleidoscope"),
    ("Frankenstein Captcha (#2604)", "captcha asking to select parts of a body to assemble a monster"),
    ("Voyager Wires (#2624)", "space probe trailing long electrical wires through the solar system"),
    ("Health Data (#2620)", "smart watch health data tracking steps and pulse"),
]


def load_comics() -> list[dict]:
    path = os.path.join("comics", "metadata.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run fetch_xkcd.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_criteria(comics: list[dict]) -> dict[str, str]:
    criteria = {}
    for c in comics:
        preview = (c.get("transcript") or c.get("explanation") or "")[:150].replace("\n", " ")
        criteria[str(c["num"])] = f"{c['title']}: {preview}"
    return criteria


async def benchmark_sequential(client: AsyncTypeSafeClient, criteria: dict[str, str], model: str = "jev-latest"):
    print(f"\n--- 1. Sequential Latency Benchmark ({model}) ---")
    latencies = []
    total_in_tokens = 0
    total_out_tokens = 0

    # Warmup
    warmup_start = time.perf_counter()
    await client.system_one(
        state={"query": "warmup"},
        questions={"warmup": Noul(instructions="Is this a test?")},
        model=model,
    )
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    print(f"Warmup ping: {warmup_ms:.1f}ms\n")

    print(f"{'Target':<30} | {'Latency':<9} | {'Top Match':<25} | {'Prob':<6} | {'Conf':<6} | {'HasMatch'}")
    print("-" * 100)

    for target_label, query in BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        resp = await client.system_one(
            state={"query": query},
            questions={
                "has_match": Noul(instructions=f'Does any XKCD comic match "{query}"?'),
                "match": Choice(instructions=f'Which XKCD comic best matches "{query}"?', criteria=criteria),
            },
            model=model,
        )
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        latencies.append(ms)

        match_ans = resp.answers["match"]
        top_cid = match_ans.choice
        conf = match_ans.confidence
        prob = match_ans.probabilities.get(top_cid, 0.0)
        has_match = resp.answers["has_match"].noul

        if resp.usage:
            total_in_tokens += resp.usage.input_tokens or 0
            total_out_tokens += resp.usage.output_tokens or 0

        print(f"{target_label:<30} | {ms:6.1f}ms  | #{top_cid:<23} | {prob:.2f}  | {conf:.2f}  | {has_match:.2f}")

    latencies.sort()
    avg_lat = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.9)]
    min_lat = latencies[0]
    max_lat = latencies[-1]

    print("-" * 100)
    print(f"Summary: Min={min_lat:.1f}ms | Avg={avg_lat:.1f}ms | P50={p50:.1f}ms | P90={p90:.1f}ms | Max={max_lat:.1f}ms")
    print(f"Tokens: {total_in_tokens} input, {total_out_tokens} output")
    return avg_lat


async def benchmark_concurrent(client: AsyncTypeSafeClient, criteria: dict[str, str], model: str = "jev-latest"):
    print(f"\n--- 2. Concurrent Throughput Benchmark ({model}) ---")
    print(f"Firing {len(BENCHMARK_QUERIES)} queries simultaneously via asyncio.gather...")

    async def single_call(target_label, query):
        t0 = time.perf_counter()
        resp = await client.system_one(
            state={"query": query},
            questions={
                "match": Choice(instructions=f'Which XKCD comic best matches "{query}"?', criteria=criteria),
            },
            model=model,
        )
        t1 = time.perf_counter()
        return (t1 - t0) * 1000, resp.answers["match"].choice

    t_start = time.perf_counter()
    results = await asyncio.gather(*[single_call(lbl, q) for lbl, q in BENCHMARK_QUERIES])
    total_time_ms = (time.perf_counter() - t_start) * 1000

    print(f"All {len(BENCHMARK_QUERIES)} queries completed in: {total_time_ms:.1f}ms")
    qps = (len(BENCHMARK_QUERIES) / (total_time_ms / 1000.0))
    print(f"Effective throughput: {qps:.1f} queries/second")
    for (lbl, _), (lat, cid) in zip(BENCHMARK_QUERIES, results):
        print(f"  {lbl:<30} -> #{cid:<6} in {lat:6.1f}ms")


async def main():
    comics = load_comics()
    print(f"Loaded {len(comics)} comics from metadata.json")
    criteria = build_criteria(comics)

    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        print("Error: TYPESAFE_API_KEY is not set.")
        sys.exit(1)

    masked_key = api_key[:10] + "..." + api_key[-4:]
    print(f"Using TypeSafe API Key: {masked_key}")

    async with AsyncTypeSafeClient() as client:
        # Benchmark with jev-latest
        await benchmark_sequential(client, criteria, model="jev-latest")
        await benchmark_concurrent(client, criteria, model="jev-latest")


if __name__ == "__main__":
    asyncio.run(main())
