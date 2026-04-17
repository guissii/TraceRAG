import sys
import time
import httpx
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_FILE = "test_docs/cours_reseaux_p1_58.txt"

def test_health():
    print("=== [1] Test du point de terminaison /health ===")
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        print("✅ Serveur RAG est en ligne!")
        print(f"  - Embedding Model: {data.get('embedding_model')}")
        print(f"  - LLM Model: {data.get('llm_model')}")
        print(f"  - Total Chunks in DB: {data.get('total_chunks')}")
        return True
    except Exception as e:
        print(f"❌ Erreur Health check: {e}")
        return False

def test_upload():
    print(f"\n=== [2] Test d'ingestion de document MASSIF ({TEST_FILE}) ===")
    file_path = Path(TEST_FILE)
    if not file_path.exists():
        print(f"⚠️ Le fichier de test {file_path} n'existe pas. Ignoré.")
        return True

    print("  - Upload en cours... (Patientez, 58 pages demandent beaucoup d'embeddings GPU)")
    start_time = time.time()
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "text/plain")}
            # 600s timeout car 58 pages à traiter
            response = httpx.post(f"{BASE_URL}/upload", files=files, timeout=600.0)
            
        response.raise_for_status()
        data = response.json()
        duration = time.time() - start_time
        print(f"✅ Document indexé avec succès en {duration:.2f}s !")
        print(f"  - Nom: {data.get('document_name')}")
        print(f"  - Nombre de chunks générés: {data.get('chunks_count')}")
        
        timings = data.get('step_timings', {})
        print(f"  - Extraction: {timings.get('extraction')} ms")
        print(f"  - Chunking: {timings.get('chunking')} ms")
        print(f"  - Embedding GPU: {timings.get('embedding')} ms")
        return True
    except Exception as e:
        print(f"❌ Erreur Upload: {e}")
        return False

def test_chat():
    print("\n=== [3] Test de génération ciblée (Ollama) ===")
    # Une question classique d'un cours de réseaux
    question = "Quelles sont les couches du modèle OSI de bas en haut et à quoi servent-elles ?"
    print(f"  - Question: '{question}'")
    
    start_time = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/chat", 
            json={
                "question": question, 
                "n_chunks": 4, # Réduit à 4 (approx. 1600 mots) pour respecter la limite GPU 4Go
                "document_name": "cours_reseaux_p1_58.txt"
            },
            timeout=300.0
        )
        response.raise_for_status()
        data = response.json()
        duration = time.time() - start_time
        
        print(f"✅ Réponse générée en {duration:.2f}s !")
        print(f"\n[Réponse Améliorée]\n{data.get('answer')}\n")
        
        sources = data.get('sources', [])
        print(f"  - Chunks (morceaux) sources récupérés: {len(sources)}")
        
        steps = data.get('pipeline_steps', [])
        for step in steps:
            print(f"  - Etape '{step.get('step')}': {step.get('duration_ms')} ms")
        
        return True
    except Exception as e:
        print(f"❌ Erreur Chat: {e}")
        return False

def main():
    print("🚀 Démarrage du Test sur Documents Massifs (58 pages)\n")
    if not test_health():
        sys.exit(1)
        
    test_upload()
    test_chat()
    
    print("\n✨ Test d'Amélioration terminé.")

if __name__ == "__main__":
    main()
