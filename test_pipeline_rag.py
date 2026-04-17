import sys
import time
import httpx
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_FILE = "test_docs/rag_architecture.txt"

def test_health():
    print("=== [1] Test du point de terminaison /health ===")
    try:
        response = httpx.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print("✅ Serveur RAG est en ligne!")
        print(f"  - Embedding Model: {data.get('embedding_model')}")
        print(f"  - LLM Model: {data.get('llm_model')}")
        print(f"  - Total Chunks in DB: {data.get('total_chunks')}")
        return True
    except httpx.ConnectError:
        print("❌ Erreur: Impossible de se connecter au serveur. Assurez-vous d'avoir lancé `.\\launch.bat`.")
        return False
    except Exception as e:
        print(f"❌ Erreur Health check: {e}")
        return False

def test_upload():
    print(f"\n=== [2] Test d'ingestion de document ({TEST_FILE}) ===")
    file_path = Path(TEST_FILE)
    if not file_path.exists():
        print(f"⚠️ Le fichier de test {file_path} n'existe pas. Ignoré.")
        return True

    print("  - Upload en cours... Cela va solliciter le GPU pour l'Embedding.")
    start_time = time.time()
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "text/plain")}
            response = httpx.post(f"{BASE_URL}/upload", files=files, timeout=60.0)
            
        response.raise_for_status()
        data = response.json()
        duration = time.time() - start_time
        print(f"✅ Document indexé avec succès en {duration:.2f}s !")
        print(f"  - Nom: {data.get('document_name')}")
        print(f"  - Nombre de chunks générés: {data.get('chunks_count')}")
        print(f"  - Temps d'embedding: {data.get('step_timings', {}).get('embedding', 'N/A')} ms")
        return True
    except httpx.ReadTimeout:
        print("❌ Erreur: Timeout lors de l'indexation. GPU est potentiellement surchargé.")
        return False
    except Exception as e:
        print(f"❌ Erreur Upload: {e}")
        return False

def test_chat():
    print("\n=== [3] Test de génération RA (Ollama) ===")
    question = "Qu'est-ce que l'architecture RAG ?"
    print(f"  - Question: '{question}'")
    
    start_time = time.time()
    try:
        response = httpx.post(
            f"{BASE_URL}/chat", 
            json={"question": question, "n_chunks": 3},
            timeout=180.0
        )
        response.raise_for_status()
        data = response.json()
        duration = time.time() - start_time
        
        print(f"✅ Réponse générée en {duration:.2f}s !")
        print(f"\n[Réponse]\n{data.get('answer')}\n")
        
        sources = data.get('sources', [])
        print(f"  - Chunks sources utilisés: {len(sources)}")
        
        steps = data.get('pipeline_steps', [])
        for step in steps:
            print(f"  - Etape '{step.get('step')}': {step.get('duration_ms')} ms")
        
        return True
    except httpx.ReadTimeout:
        print("❌ Erreur: Timeout de génération Ollama. Le modèle est-il bien chargé/le serveur est-il bloqué?")
        return False
    except Exception as e:
        print(f"❌ Erreur Chat: {e}")
        return False

def main():
    print("🚀 Démarrage de l'Audit & Test du Serveur TraceRAG\n")
    if not test_health():
        sys.exit(1)
        
    test_upload()
    test_chat()
    
    print("\n✨ Audit et Test terminés.")

if __name__ == "__main__":
    main()
