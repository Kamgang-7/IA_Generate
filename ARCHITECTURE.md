# 🏗️ Architecture Technique : SmartPDF-RAG

Ce document détaille la conception interne, la gestion des données et les choix stratégiques du système de **Retrieval-Augmented Generation (RAG)** mis en œuvre dans ce projet.

---

## 1. Pipeline de Traitement des Requêtes (Workflow)

Le système suit un processus séquentiel pour transformer une question utilisateur en une réponse précise, sourcée et vérifiable.



### 🔄 Étape 1 : Contextualisation (Query Rewriting)
Lorsqu'un utilisateur pose une question au milieu d'une conversation (ex: *"Peux-tu m'en donner plus de détails ?"*), la question seule est souvent trop vague pour une recherche efficace.
* **Action** : Le système utilise le `REWRITE_PROMPT` pour fusionner l'historique et la question en une **requête autonome**.
* **Objectif** : Maximiser la pertinence de la recherche lexicale en identifiant les mots-clés exacts.

### 🔍 Étape 2 : Recherche Documentaire (Retrieval)
La requête reformulée est soumise au moteur de recherche local.
* **Algorithme** : **BM25Okapi** (Ranking statistique basé sur la fréquence inverse des documents).
* **Extraction** : Le système récupère les **$k=4$** fragments (chunks) ayant le score de pertinence le plus élevé.
* **Sources** : Chaque fragment conserve ses métadonnées (nom du fichier, numéro de page).

### ✍️ Étape 3 : Synthèse et Réponse (Generation)
Le LLM (Google Gemini 1.5 Flash) reçoit un prompt final "augmenté" contenant les extraits trouvés.
* **Contrainte** : L'IA a l'ordre formel de ne pas inventer d'informations si la réponse ne figure pas dans le contexte fourni (lutte contre les hallucinations).

---

## 2. Structure et Persistance des Données

Pour garantir un démarrage instantané sans re-parser les PDF à chaque lancement, le projet utilise une indexation locale intelligente dans le dossier `bm25_index/`.



### 📄 Le Manifeste (`manifest.json`)
Ce fichier agit comme un système de contrôle de version pour les documents.
* **Fingerprint (SHA-256)** : Une empreinte numérique unique générée à partir du nom, de la taille et de la date de modification de tous les fichiers du dossier `PDF/`.
* **Logique de Cache** : Si le fingerprint calculé au démarrage est identique à celui stocké, le système charge l'index existant. Sinon, il déclenche une reconstruction automatique.

### 📦 Le Stockage (`store.json`)
Contient la "mémoire" textuelle du système :
* **Texts** : Les paragraphes découpés (chunks) de 1000 caractères.
* **Metas** : Les informations sources (source, page) liées à chaque paragraphe pour assurer la traçabilité.

---

## 3. Stratégie de Prompt Engineering

L'efficacité du système repose sur deux prompts piliers :

### A. Le Prompt de Reformulation (`REWRITE_PROMPT`)
* **Rôle** : "Nettoyeur" de contexte.
* **Logique** : Il transforme une intention humaine (parfois vague) en une requête optimisée pour un algorithme statistique.

### B. Le Prompt Système (`MANUAL_PROMPT_TEMPLATE`)
* **Instruction d'Honnêteté** : *"Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas."*
* **Structuration** : Force l'IA à utiliser des listes à puces pour la clarté.
* **Ancrage** : *"Utilisez uniquement les morceaux de contexte suivants."*

---

## 4. Choix de l'Algorithme : BM25 vs Vector Search

[Image comparing BM25 keyword matching vs vector embedding semantic search]

| Caractéristique | BM25 (Notre choix) | Recherche Vectorielle (FAISS) |
| :--- | :--- | :--- |
| **Type de recherche** | Lexicale (mots-clés exacts). | Sémantique (sens global). |
| **Précision** | Excellente sur les noms propres, codes et termes techniques. | Meilleure sur les concepts et les synonymes. |
| **Infrastructure** | **Zéro coût**, calcul local ultra-léger. | Nécessite des modèles d'embeddings (payants ou lourds). |
| **Maintenance** | Fichiers JSON simples. | Nécessite une gestion de base de données de vecteurs. |

---

## 5. Monitoring avec Langfuse

L'architecture intègre nativement le traçage via Langfuse pour mesurer :
1.  **La Latence** : Temps de recherche vs temps de génération.
2.  **Le Coût** : Consommation de tokens Gemini en temps réel.
3.  **Le Débogage** : Visualisation de l'étape de reformulation pour ajuster les prompts.