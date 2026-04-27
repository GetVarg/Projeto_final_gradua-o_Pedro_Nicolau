from datasets import load_dataset

# 1. Carrega o dataset do Hugging Face
ds = load_dataset("Smarneh/NIT")

# 2. Salva no formato JSON Lines
# Se o dataset tiver divisões (train, test), você deve salvar cada uma
ds["train"].to_json("data/meu_dataset_nit.jsonl")

print("Dataset salvo com sucesso em meu_dataset_nit.jsonl")