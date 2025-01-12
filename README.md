# Multimodal-RAG

---

## Repository Structure

- `app.py`  
  The main application file for launching the service.

- `requirements.txt`  
  A list of dependencies required for the project to work.

- `data/`  
  Directory containing data for indexing and searching.
  
- `src/`  
  Project source code.
  - `llm/`  
    Modules for working with large language models.
  - `retrievers/`  
    Modules for extracting relevant information.
  - `utils.py`  
    Helper functions and utilities.

---

## Running

To run the project, follow these steps:

1. **Install poppler**
    ```bash
    !sudo apt-get install -y poppler-utils
    ```
    
2. **Create and activate a virtual environment:**
  
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
  
3. **Install dependencies:**

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    pip install -r requirements.txt
    ```
4. **Add a `.env` file to the root of the project, contact us, we'll provide a key) tg @umbilnm**

5. **Run the application:**
    
    ```bash
    streamlit run app.py
    ```
    
6. **Access the application:**
    
    Open a web browser and navigate to `http://localhost:8000`.

---

## Build and Launch Time

1. **Installing dependencies:**  
   - Takes about **2 minutes**.

2. **Launching the application:**  
   - Instant launch after installing dependencies.
---

## Solution Overview

Our project includes the following key components:

**Data Indexing**
- Using **FAISS** for efficient embedding-based search.
- Storing metadata and embeddings for quick access.

**Working with LLM**
- Integration with Pixtral-12b for generating responses.

## Hypothesis: Combining Textual and Visual Embeddings to Enhance Multimodal Search

The main hypothesis of our solution is the assumption that combining embeddings from textual and visual modalities significantly improves the accuracy and quality of multimodal search. We believe that integrating data from different types of sources (text descriptions and images) provides a deeper understanding of context and increases the relevance of the results.

---

### Our Approach: Strategies for Implementing the Hypothesis

To test this hypothesis, we developed several strategies that allow for different combinations of textual and visual embeddings. Each of them represents a part of the overall approach to solving the multimodal search problem.

#### 1. **SummaryEmb**

This strategy extracts images using textual embeddings obtained from image descriptions (summary) via Pixtral-12b. It helps account for the textual context of images but does not directly utilize visual information.

#### 2. **ColQwen**

Visual embeddings obtained through the ViT model are used to extract images relevant to the query. This strategy allows for considering exclusively the visual characteristics of images.

#### 3. **Intersection**

This strategy combines the results of both modalities by intersecting images found by textual and visual embeddings. This allows for considering both textual and visual relevance, which is important for multimodal queries.

#### 4. **ColQwen+SummaryEmb**

This strategy combines the top results of textual and visual embeddings, selecting the most relevant images from both approaches. It helps effectively solve tasks that require multimodal analysis.
