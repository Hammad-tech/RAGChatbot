# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to chat with their documents using OpenAI's language models.

## Features

- Document upload and processing
- OpenAI integration
- User authentication
- Chat history tracking
- Admin functionality for database documents
- Multi-language support

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/RAGChatbot.git
cd RAGChatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Set up the database:
```bash
python RAG_app.py
```
This will create the necessary database and directory structure.

## Directory Structure

```
RAGChatbot/
├── data/
│   ├── database/          # Place your PDF documents here
│   ├── pretrained_vector_store/  # Generated embeddings will be stored here
│   ├── tmp/              # Temporary files
│   └── vector_stores/    # User-uploaded document embeddings
├── RAG_app.py           # Main application
├── generate_embeddings.py # Script to generate embeddings
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## Usage

1. Start the application:
```bash
streamlit run RAG_app.py
```

2. For database documents:
   - Place your PDF files in the `data/database` directory
   - Run the embeddings generation script:
   ```bash
   python generate_embeddings.py
   ```
   - The embeddings will be stored in `data/pretrained_vector_store`

3. For individual documents:
   - Use the document upload feature in the sidebar
   - The embeddings will be generated automatically

## Notes

- The vector store files are not included in the repository due to their size
- You need to generate embeddings locally after cloning the repository
- Make sure you have sufficient OpenAI API credits for generating embeddings

## License

This project is licensed under the MIT License.
