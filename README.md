# 🎨 ArtRag

**ArtRag** is a simple Streamlit application for exploring artworks and artists.  
It allows you to load sample artwork data or a saved embedding index and ask questions about art.  
This project is a starting point for building a retrieval-augmented generation (RAG) system for the arts.

---

## ✨ Features

- Load sample data about artworks and artists
- (Planned) Load a preserved local embedding index for semantic search
- Ask free-text questions about artworks or artists
- Clean and minimal Streamlit interface
- Easily extendable for advanced search or question answering systems

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/marleentheyoung/ArtRag.git
cd ArtRag
```

### 2. Create and activate a virtual environment (recommended)

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):

```bash
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install the required Python packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

In the project directory, run the following command:

```bash
streamlit run app.py
```

Then open your browser at:  
[http://localhost:8501](http://localhost:8501)

---

## 💬 Usage

1. Use the sidebar to:
   - **Load Sample Data** (basic artwork data)
   - **Load Embedding Index** (planned feature)

2. Ask a question about an artwork or artist in the input box.

3. Results will be shown below (search functionality coming soon).

---

## 🚀 Planned Features

- Add semantic search over artwork descriptions
- Integrate with external APIs like Wikipedia or WikiArt
- Display artwork images alongside descriptions
- Save and reload custom embedding indexes

---

