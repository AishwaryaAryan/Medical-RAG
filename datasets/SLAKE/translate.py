import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

translator = GoogleTranslator(source='zh-CN', target='en')

files = [
    "KG/disease.csv",
    "KG/organ.csv",
    "KG/organ_rel.csv"
]

for file in files:
    print(f"Translating {file}...")

    df = pd.read_csv(file)

    for col in df.columns:
        translated = []

        for value in tqdm(df[col].astype(str)):
            try:
                translated.append(
                    translator.translate(value)
                )
            except:
                translated.append(value)

        df[col] = translated

    output_file = file.replace(".csv", "_english.csv")
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("All files translated.")