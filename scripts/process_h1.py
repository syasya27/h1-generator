import os
import re
import csv
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

from openai import AsyncOpenAI

INPUT_CSV = os.getenv("INPUT_CSV", "data/h1_input.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "data/h1_output.csv")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "data/checkpoints")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
CONCURRENCY = int(os.getenv("CONCURRENCY", "50"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "500"))

EXCEL_ERRORS = {"#NAME?", "#VALUE!", "#REF!", "#N/A", "#DIV/0!"}

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
sem = asyncio.Semaphore(CONCURRENCY)

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_ok_marker(text: str) -> bool:
    return text.strip().lower() == "ok"


def is_excel_error(text: str) -> bool:
    return text.strip().upper() in EXCEL_ERRORS


def is_pipe_garbage(text: str) -> bool:
    text = text.strip()
    return bool(text) and set(text) == {"|"}


def has_real_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿĀ-žА-Яа-яЁё]", text or ""))


def products_missing_or_garbage(products: str) -> bool:
    p = normalize(products)
    if not p:
        return True
    if is_pipe_garbage(p):
        return True
    if not has_real_text(p):
        return True
    return False


def fallback_old_or_empty(old_h1: str) -> str:
    return normalize(old_h1)


def build_prompt(products: str, current_h1: str, mode: str) -> str:
    """
    mode:
      - generate
      - validate
    """
    rules = """
You are a senior SEO editor for ecommerce category pages.

Your task is to produce the best possible category H1.

Core rules:
1. The H1 must reflect the real product group from the products list.
2. Keep it short, natural, and category-like.
3. Do NOT add redundant qualifiers that are already implied by the product type.
   Example: do NOT write "Condoms for Adults" because condoms are already for adults.
4. Do NOT add marketing words like:
   - best
   - cheap
   - low price
   - premium
   - original
   - sale
5. Do NOT invent extra attributes unless clearly supported by the products.
6. Prefer a clean marketplace/category style.
7. No punctuation decoration, no emojis.
8. Return ONLY the final H1 text, with no explanation.

How to reason:
- Detect the main product type from the products.
- Remove redundant, overly specific, or weird wording.
- If the current H1 is already correct and natural, keep it.
- If the current H1 is missing, broken, or semantically bad, create a better one.
"""

    if mode == "generate":
        task = """
Task: Generate a clean category H1 based on the products list.
"""
    else:
        task = """
Task: Validate the current H1 against the products list.
If it is good, keep it.
If it is redundant, unnatural, too broad, too narrow, or semantically wrong, fix it.
"""

    prompt = f"""
{rules}

{task}

Products list:
{products}

Current H1:
{current_h1 if current_h1 else "[EMPTY]"}

Return only the final H1.
"""
    return prompt.strip()


async def call_model(products: str, current_h1: str, mode: str) -> str:
    prompt = build_prompt(products=products, current_h1=current_h1, mode=mode)

    async with sem:
        response = await client.responses.create(
            model=MODEL,
            input=prompt,
        )

    # SDK shape can vary slightly by version, so keep extraction defensive
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    # Fallback extraction
    try:
        parts = []
        for item in response.output:
            if getattr(item, "type", "") == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", "") == "output_text":
                        parts.append(content.text)
        return " ".join(parts).strip()
    except Exception:
        return ""


async def process_row(row: Dict[str, str], idx: int) -> Dict[str, str]:
    url = normalize(row.get("url"))
    old_h1 = normalize(row.get("h1_old"))
    products = normalize(row.get("products"))
    new_h1 = normalize(row.get("h1_new"))

    result = dict(row)
    result["decision"] = ""
    result["h1_final"] = ""
    result["notes"] = ""

    # 1) OK -> old
    if is_ok_marker(new_h1):
        result["decision"] = "use_old_because_ok"
        result["h1_final"] = fallback_old_or_empty(old_h1)
        if not result["h1_final"]:
            result["notes"] = "old_h1_empty_after_ok"
        return result

    # 2) products empty/garbage -> old
    if products_missing_or_garbage(products):
        result["decision"] = "use_old_because_no_products"
        result["h1_final"] = fallback_old_or_empty(old_h1)
        if not result["h1_final"]:
            result["notes"] = "old_h1_empty_and_no_products"
        return result

    # 3) h1_new missing or excel error -> generate
    if not new_h1 or is_excel_error(new_h1):
        generated = await call_model(products=products, current_h1="", mode="generate")
        result["decision"] = "generated_because_missing_or_error"
        result["h1_final"] = generated
        return result

    # 4) validate existing h1_new
    validated = await call_model(products=products, current_h1=new_h1, mode="validate")
    result["decision"] = "validated_existing_h1"
    result["h1_final"] = validated
    return result


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(rows: List[Dict[str, str]], chunk_index: int) -> None:
    cp_path = Path(CHECKPOINT_DIR) / f"checkpoint_{chunk_index:05d}.csv"
    write_csv(str(cp_path), rows)


async def main():
    rows = read_csv(INPUT_CSV)
    processed_rows: List[Dict[str, str]] = []

    tasks = []
    for idx, row in enumerate(rows, start=1):
        tasks.append(process_row(row, idx))

    chunk: List[Dict[str, str]] = []
    chunk_index = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        processed_rows.append(result)
        chunk.append(result)

        if len(chunk) >= SAVE_EVERY:
            chunk_index += 1
            save_checkpoint(chunk, chunk_index)
            chunk = []

    if chunk:
        chunk_index += 1
        save_checkpoint(chunk, chunk_index)

    # preserve original order by url/h1_old/h1_new/products order is not guaranteed from as_completed
    # easiest stable merge: process again by mapping tuple -> first result
    # for duplicates we use queue behavior
    from collections import defaultdict, deque

    buckets = defaultdict(deque)
    for row in processed_rows:
        key = (
            normalize(row.get("url")),
            normalize(row.get("h1_old")),
            normalize(row.get("products")),
            normalize(row.get("h1_new")),
        )
        buckets[key].append(row)

    ordered_rows = []
    for row in rows:
        key = (
            normalize(row.get("url")),
            normalize(row.get("h1_old")),
            normalize(row.get("products")),
            normalize(row.get("h1_new")),
        )
        ordered_rows.append(buckets[key].popleft())

    write_csv(OUTPUT_CSV, ordered_rows)
    print(f"Done. Wrote {len(ordered_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
