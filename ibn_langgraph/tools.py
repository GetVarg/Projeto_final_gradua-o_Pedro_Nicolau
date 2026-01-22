import json
import os
from pathlib import Path

# --- FUNÇÃO QUE O GRAFO VAI USAR ---
def load_topologia(path="data/topologia_augmented.json") -> dict:
    """
    Lê o arquivo JSON gerado pelo Mininet para fornecer contexto à LLM.
    """
    caminho = Path(path)
    if not caminho.exists():
        print(f"Aviso: Arquivo {path} não encontrado. Certifique-se de que a topologia foi exportada.")
        return {}
    
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler a topologia: {e}")
        return {}