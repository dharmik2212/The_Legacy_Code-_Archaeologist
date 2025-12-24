# from langchain_community

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.output_parsers import StrOutputParser
import requests
from dotenv import load_dotenv

load_dotenv()

class GithubLoaderWithHistory(GithubFileLoader):
    def load(self):
        # 1. Let the original loader get the file content
        docs = super().load()
        
        # 2. Automatically fetch history for each file found
        for doc in docs:
            filename = doc.metadata['source']
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
    
embeddings= HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)                                              
llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN'),
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
REPO_OWNER = "renish-1111"
REPO_NAME = "ToDo"
target_filename = "index.html"  # <--- CHANGE THIS to the exact file you want

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
    language=Language.HTML,
    chunk_size=500,
    chunk_overlap=50
)


splited_text = splitter.split_documents(raw_docs)

vectorstore = FAISS.from_documents(splited_text, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 4, "fetch_k": 20}
)

def format_docs(docs):
    formatted_output = ""
    for doc in docs:
        source_file = doc.metadata.get('source', 'Unknown File')
        git_history = doc.metadata.get('history', 'No history found.')
        formatted_output += f"\nFILE: {source_file}\n"
        formatted_output += f"--- GIT COMMIT HISTORY (Latest first) ---\n"
        formatted_output += git_history
        formatted_output += f"\n--- CODE CONTENT ---\n"
        formatted_output += doc.page_content
        formatted_output += "\n" + "="*40
    return formatted_output

prompt= PromptTemplate(
    template="""
    You are a senior developer analyzing code.
    You have been given code snippets and their corresponding Git commit history.
    
    INSTRUCTIONS:
    1. Look at the 'GIT COMMIT HISTORY' section to see WHO made changes and WHEN.
    2. The first entry in the history is the LATEST modification.
    3. If asked about a specific class or line, match it to the history provided.
    
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
res=chain.invoke('what happens if I remove   <button onClick={(e)=>handleEdit> i sure this line or code commited so give answer based on it')
print(res)


