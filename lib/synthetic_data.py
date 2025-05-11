import os, json, re, openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SYSTEM_PROMPT = """You are a Russian-to-English translator specialized in the field of biotechnology and medicine.
Given a Russian text and it's translation, you will generate a similar text in Russian and it's English translation.
The texts should be similar in meaning and length to the original ones, as they will be used to train a machine learning model.
Make sure that the generated texts are not identical to the original text, but still convey the same meaning, use the same technical terms,
 and are similar in length."""

EXAMPLE_INPUT = """
[Russian]: Вакцинация против COVID-19 является важным шагом в борьбе с пандемией.
[Translation]: Vaccination against COVID-19 is an important step in the fight against the pandemic.
"""

EXAMPLE_OUTPUT = """
[Russian]: Вакцинация от COVID-19 - это ключевой шаг в борьбе с пандемией.
[Translation]: Vaccination against COVID-19 is a key step in the fight against the pandemic.
"""



def generate_pair(
    client: openai.Client,
    ru: str,
    en: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 1.0,
    max_tokens: int = 300,
    idx = 0,
) -> tuple[str, str]:
    usr = f"[Russian]: {ru}\n[Translation]: {en}"
    rsp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "assistant", "content": EXAMPLE_INPUT},
                  {"role": "user", "content": EXAMPLE_OUTPUT},
                  {"role": "user", "content": usr}],
        temperature=temperature,
        max_tokens=max_tokens,
    ).choices[0].message.content.strip()
    
    ru_new, en_new = re.findall(r"\[Russian\]: (.+?)\n\[Translation\]: (.+)", rsp)[0]
    return {
        "ru": ru_new,
        "en": en_new,
        "idx": idx,
    }

def expand_dataset_parallel(
    pairs: list[tuple[str, str]],
    out_file: str,
    api_key: str | None = None,
    workers: int = 8,
    model: str = "gpt-4.1-mini",
    temperature: float = 1,
    max_tokens: int = 300,
    new_samples_per_pair: int = 1,
) -> None:
    client = openai.Client(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, (ru, en) in tqdm(enumerate(pairs), desc="Generating pairs", total=len(pairs)):
            for _ in range(new_samples_per_pair):
                futures.append(executor.submit(generate_pair, client, ru, en, model, temperature, max_tokens))

        with open(out_file, "w") as f:
            for future in tqdm(as_completed(futures), desc="Writing pairs", total=len(futures)):
                try:
                    pair = future.result()
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error generating pair: {e}")