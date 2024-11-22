📄 Document Q&A App

This app allows you to ask questions directly from your uploaded documents using OpenAI GPT-4 and LangChain. Designed for researchers, students, and professionals, it efficiently processes PDFs, TXT, and DOCX files to provide accurate, context-based answers.

Features

- Document Upload: Supports PDF, TXT, and DOCX formats.
- Vector Embeddings: Creates a vectorized representation of your documents using FAISS.
- Question Answering: GPT-4 answers your queries based on the document content.
- Document Preview: Review the content of uploaded documents before embedding.
- Save & Load Vector Store: Reuse vectorized data across sessions.
- Similarity Search: Displays document chunks relevant to your query.
- Response Time Tracking: Know how fast your queries are processed.
- Usage

* Place your documents in the ./us_census folder.
* Run the app locally using streamlit run app.py.
* Generate embeddings from the sidebar.
* Ask your questions in the input box and get instant answers!

  
Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API Key
- 
Check out the full documentation in the repo. Contributions are welcome! 😊
