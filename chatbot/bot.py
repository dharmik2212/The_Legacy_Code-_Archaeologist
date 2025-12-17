# from langchain_community
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.output_parsers import StrOutputParser
import os
import requests
from dotenv import load_dotenv
from torch import embedding
load_dotenv()

class GithubLoaderWithHistory(GithubFileLoader):
    def load(self):
        # 1. Let the original loader get the file content
        docs = super().load()
        
        # 2. Automatically fetch history for each file found
        for doc in docs:
            filename = doc.metadata['source']
            print(f"🤖 Automatically fetching history for: {filename}...")
            
            # Call GitHub API for history
            url = f"https://api.github.com/repos/{self.repo}/commits"
            params = {"path": filename, "per_page": 5} # Get last  5 commits
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(url, headers=headers, params=params)
            
            history_str = ""
            if response.status_code == 200:
                for commit in response.json():
                    author = commit['commit']['author']['name']
                    date = commit['commit']['author']['date'].split("T")[0]
                    msg = commit['commit']['message']
                    history_str += f"- [{date}] {author}: {msg}\n"
            
            # 3. Inject history directly into the Document
            doc.metadata['history'] = history_str
            
        return docs
    
embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN'),
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
REPO_OWNER = "sanket-sakariya"
REPO_NAME = "URL-SHORTENER"
target_filename = "app.py"  # <--- CHANGE THIS to the exact file you want

# print(f"dt Fetching ONLY {target_filename}...")

loader = GithubLoaderWithHistory(
    repo=f"{REPO_OWNER}/{REPO_NAME}",
    access_token=os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'),
    branch="main",
    # 2. FILTER: This lambda function returns True ONLY if the filename matches
    file_filter=lambda file_path: file_path.endswith(target_filename)
)

raw_docs = loader.load()

if not raw_docs:
    print(f"❌ Error: Could not find '{target_filename}' in the repo.")
    exit()

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

print("✂️ Splitting code...")
splited_text = splitter.split_documents(raw_docs)

vectorstore = FAISS.from_documents(splited_text, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 4, "fetch_k": 20}
)

def format_docs(docs):
    formatted_output = ""
    for doc in docs:
        formatted_output += f"\n--- RELEVANT CODE SNIPPET ---\n"
        formatted_output += doc.page_content
        formatted_output += f"\n\n--- HISTORY FOR THIS FILE ---\n"
        formatted_output += doc.metadata.get('history', 'No history found.')
        formatted_output += "\n" + "="*20
    return formatted_output

prompt= PromptTemplate(
    template="""
    You are a senior developer analyzing code.
    Use the following retrieved context (Code Snippets + Git History) to answer the question.
    
    CONTEXT:
    {context}
    
    QUESTION: {topic}
    
    ANSWER:
    """,
    input_variables=['topic', 'context']
)
parser= StrOutputParser()
chain=chain = (
    {"context": retriever | format_docs, "topic": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
res=chain.invoke('Who modified the Shortener class and what happens if I remove it?')
print(res)