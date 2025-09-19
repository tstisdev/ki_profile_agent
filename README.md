# Profil KI-Bot

## Ziel:
Geplant war, dass wir diesen Bot mit Azure und Sharepoint verbinden, sodass er in Teams eingebunden werden kann.
Azure sollte das AzureOpenAI LLM für Generation und Embeddings bereitstellen.

Da wir mit den Berechtigungen für Sharepoint nicht weitergekommen sind, funktioniert es gerade fast Komplett Lokal.
Für Generation wird noch OpenAI mit einem OpenAI API Key benutzt. Bzw. wurde, da dieser mein privater Key war. 
Dieser muss einfach in der .env geändert werden.

In der Docker-compose.yml wird ein lokaler Ordner referenziert, wo die Profil PDFs liegen müssen.

## Setup / Prerequisites
 - [x] Docker Desktop
 - [x] OpenAI API Key (man kann auch, wenn man Azure hat, einfach das einbauen, sind 2-3 Zeilen Code die man ändern muss im vector_store.py)
 - [x] ODER: man startet ein lokales LLM (Ollama, vLLM (Linux only)) und bindet den Service aus der docker-compose.yml wieder ein

## Build / Starten

1. App bauen: ``docker compose build rap_app``
2. Starten: ``docker compose run --rm -it rag_app python src/main.py``
   Hier wird in der Console die App ausgeführt, nach ein paar Logs kommt "Enter Question: ", dann kann man die Profile durchsuchen
3. Wenn man PDFs, Models / Dimension / chunk sizes ändert und man schon eine DB mit Embeddings hat muss man 
``docker compose run --rm -it rag_app python -c "from src.rag_pipeline import RAGPipeline; p=RAGPipeline(); p.initialize(force_rebuild=True); print('reindexed')"
`` ausführen.

## Beispiel Output

![img.png](img.png)


## Warning!
    Ich habe es bisher mit nur 2 PDFs getestet, sollte der Output mit mehreren nicht zufriedenstellend sein, muss man entweder den PROMPT anpassen, oder TOP_K erhöhen.