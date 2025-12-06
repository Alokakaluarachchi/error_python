# Error Log Helper

A professional web application that analyzes error logs and suggests relevant Stack Overflow solutions using AI-powered embeddings and semantic similarity.

## Features

- **Smart Error Extraction**: Uses embeddings and cosine similarity to identify the most relevant error lines from logs
- **Local Knowledge Base**: Pre-processed Q&A database with embeddings for fast matching
- **Stack Overflow Integration**: Fetches relevant solutions directly from Stack Overflow API
- **Professional UI**: Clean, modern dark-themed interface
- **History Tracking**: Stores recent queries in JSON format

## Tech Stack

- **Backend**: Python 3.8+ with FastAPI
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **AI/ML**: sentence-transformers (all-MiniLM-L6-v2 model)
- **Storage**: JSON files for data persistence

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - FastAPI (web framework)
   - Uvicorn (ASGI server)
   - sentence-transformers (embeddings)
   - numpy (numerical operations)
   - requests (HTTP client)

## Running the Application

1. **Start the backend server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

2. **Open the frontend**:
   - Simply open `index.html` in your web browser
   - Or access via `http://localhost:8000` if you configure FastAPI to serve static files

## Usage

1. **Paste your error log** into the textarea
2. **Click "Find suggestions"** (or press `Ctrl/Cmd + Enter`)
3. **View results**:
   - Top error lines extracted from your log
   - Local knowledge matches (from `data/seed_answers.json`)
   - Stack Overflow links with scores and answer counts

## Project Structure

```
ops_test/
├── main.py                 # FastAPI backend server
├── index.html             # Frontend web page
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/
    ├── seed_answers.json      # Q&A knowledge base
    ├── seed_embeddings.json   # Generated embeddings (auto-created)
    └── history.json           # Query history
```

## API Endpoints

### `POST /analyze`
Analyzes an error log and returns suggestions.

**Request Body**:
```json
{
  "log": "Your error log text here...",
  "top_k": 8
}
```

**Response**:
```json
{
  "query": "extracted keywords",
  "keywords": ["keyword1", "keyword2"],
  "highlights": ["error line 1", "error line 2"],
  "local_matches": [...],
  "results": [...]
}
```

### `GET /healthz`
Health check endpoint.

## Configuration

- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Backend URL**: Defaults to `http://localhost:8000` (can be changed in `index.html`)
- **Port**: 8000 (configurable in uvicorn command)

## Adding Custom Q&A

Edit `data/seed_answers.json` to add your own issue descriptions and answers:

```json
[
  {
    "issue": "Error description",
    "answer": "Solution text",
    "tags": ["python", "npm"]
  }
]
```

The embeddings will be automatically regenerated on the next analysis.

## Notes

- First run will generate embeddings from `seed_answers.json` (may take a few seconds)
- Requires internet connection for Stack Overflow API calls
- All data is stored locally in JSON files

## License

MIT

