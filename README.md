# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to chat with their documents using various LLM providers.

## Features

- Support for multiple LLM providers (OpenAI, Google Generative AI, HuggingFace)
- Document upload and processing
- Pre-trained knowledge base integration
- User authentication and credit system
- Admin functionality for database management
- Chat history tracking and export
- Multi-language support

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Hammad-tech/RAGChatbot.git
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

4. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

5. Run the application:
```bash
streamlit run RAG_app.py
```

## Usage

1. Register a new account or login with the admin credentials:
   - Username: admin
   - Password: admin123

2. Select your preferred LLM provider and enter the corresponding API key

3. Upload documents or use the pre-trained knowledge base

4. Start chatting with your documents!

## Project Structure

```
RAGChatbot/
├── RAG_app.py           # Main application file
├── requirements.txt     # Python dependencies
├── data/               # Data directories
│   ├── database/       # Database documents
│   ├── initial_training/ # Initial training data
│   ├── pretrained_vector_store/ # Pre-trained vector store
│   └── tmp/           # Temporary files
└── README.md          # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
