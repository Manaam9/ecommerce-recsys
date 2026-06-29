import json
import pickle
import pandas as pd
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

with open("models/inference_assets.pkl", "rb") as f:
    assets = pickle.load(f)

df = assets["item_features_df"].to_pandas()
tree = pd.read_csv("data/raw/category_tree.csv")

all_cats = df.groupby("categoryid").agg(
    items=("itemid", "count"),
    avg_buy_rate=("item_buy_rate", "mean"),
    avg_cart_rate=("item_cart_rate", "mean"),
).reset_index()
all_cats = all_cats.merge(tree, on="categoryid", how="left")
all_cats = all_cats.sort_values("items", ascending=False)

print(f"Всего категорий: {len(all_cats)}")

def generate_batch(batch):
    lines = "\n".join([
        "id={}, items={}, buy_rate={:.3f}, cart_rate={:.3f}, parent={}".format(
            int(row["categoryid"]),
            int(row["items"]),
            row["avg_buy_rate"],
            row["avg_cart_rate"],
            int(row["parentid"]) if pd.notna(row["parentid"]) else "root"
        )
        for _, row in batch.iterrows()
    ])

    prompt = (
        "Это RetailRocket — российский e-commerce датасет.\n"
        "Придумай реалистичные русские названия категорий товаров.\n\n"
        "Правила:\n"
        "- high buy_rate (>0.015) — дорогие/нишевые товары\n"
        "- high cart_rate (>0.04) — популярные товары\n"
        "- parent=root — корневая категория верхнего уровня\n"
        "- названия кратко: 2-4 слова\n\n"
        "Категории:\n"
        + lines +
        '\n\nВерни ТОЛЬКО валидный JSON без markdown:\n{"id": "название", ...}'
    )

    resp = client.chat.completions.create(
        model="qwen3.5-9b-sushi-coder-rl",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


BATCH_SIZE = 50
result = {}
batches = [all_cats.iloc[i:i+BATCH_SIZE] for i in range(0, len(all_cats), BATCH_SIZE)]

for i, batch in enumerate(batches):
    print(f"Батч {i+1}/{len(batches)} ({len(batch)} категорий)...")
    try:
        names = generate_batch(batch)
        result.update(names)
        print(f"  OK: {len(names)} названий")
    except Exception as e:
        print(f"  ERR: {e}")
        for _, row in batch.iterrows():
            cid = str(int(row["categoryid"]))
            result[cid] = f"Категория {cid}"

with open("models/category_names.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\nГотово: {len(result)} категорий")
print(f"Категория 819: {result.get('819', 'не найдена')}")
print(f"Категория 491: {result.get('491', 'не найдена')}")
