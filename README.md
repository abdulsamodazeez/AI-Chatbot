# AI Chatbot 

This project is a web application built with Streamlit that allows users to upload a PDF, extract its content, and interact with an AI chatbot. The AI, powered by OpenAI's GPT models, can use the extracted PDF content to answer user queries in a Retrieval-Augmented Generation (RAG) setup.

## Features
- PDF Upload and Extraction: Users can upload PDF files, and the application extracts text from these PDFs.
- Chat Interface: A user-friendly chat interface to interact with the AI.
- RAG Integration: Combines user queries with the context from the uploaded PDF to provide more relevant responses.
- Session Management: Maintains the history of the conversation across interactions.

## Requirements
- Python 3.7+
- Streamlit
- OpenAI API client
- PyPDF2
- An OpenAI API key

## Installation

1. Clone the repository:

```bash

git clone https://github.com/your-username/pdf-chat-app.git
cd pdf-chat-app
```
2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash

pip install -r requirements.txt
```

4. Set the OpenAI API key:

- Create a file named .env in the root directory of the project.
- Add your OpenAI API key to this file:
  

```env
API_KEY_="your_openai_api_key"
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```
2. Open your web browser and go to http://localhost:8501.

3. You can chat with the chatbot directly or Use the sidebar to upload a PDF file. The extracted text from the PDF will be used as context for the chatbot.

4. Interact with the chatbot by typing your questions in the input box and pressing Enter. The chatbot will respond using the context from the PDF.

## Project Structure

- app.py: The main Streamlit application file.
- requirements.txt: Lists the dependencies required for the project.
- .env: Contains environment variables like the OpenAI API key.

## Code Explanation

**Initialization**

The OpenAI client is initialized with the API key and base URL:

```python
openai = OpenAI(
    api_key=os.environ.get("API_KEY_"),
    base_url="https://api.deepinfra.com/v1/openai"
)
```
**Extract Text from PDF**
The extract_text_from_pdf function uses PyPDF2 to read the PDF file and extract text:

```python
def extract_text_from_pdf(pdf_file):
    reader = PdfFileReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text
```

**Chatbot Interaction**
The get_openai_response function sends the chat history to the OpenAI API and retrieves the response:

```python
def get_openai_response(message_history):
    chat_completion = openai.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=message_history
    )
    return chat_completion.choices[0].message.content
```

**PDF Context in RAG**
The get_rag_response function augments user queries with PDF content to improve response relevance:

```python
def get_rag_response(user_query, pdf_text):
    augmented_input = f"{user_query}\n\nContext from PDF:\n{pdf_text[:1000]}"  # Limiting context length
    message_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": augmented_input}
    ]
    return get_openai_response(message_history)
```

**Streamlit Interface**
The main Streamlit interface handles user interactions and displays the chat history:

```python
st.text_input("", key="user_input", on_change=send_message, value="", placeholder="Ask something...")
```

**Customization**

- Chatbot Model: Change the model by updating the model parameter in the get_openai_response function.
- PDF Processing: Enhance the PDF extraction logic for better handling of various PDF formats.
- Styling: Customize the chat interface by modifying the CSS styles in the st.markdown section.

## Contributing
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -am 'Add new feature').
5. Push to the branch (git push origin feature-branch).
6. Create a new Pull Request.
