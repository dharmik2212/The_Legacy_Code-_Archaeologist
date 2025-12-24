# from langchain_community

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.output_parsers import StrOutputParser
import requests
from dotenv import load_dotenv

load_dotenv()

class GithubLoaderWithBlame(GithubFileLoader):
    def load(self):
        docs = super().load()
        for doc in docs:
            filename = doc.metadata.get('source', '')
            print(f"🔦 Digging into Git Blame for: {filename}...")
            
            # GraphQL Query for precise line-by-line ownership
            query = """
            query($owner: String!, $repo: String!, $path: String!) {
              repository(owner: $owner, name: $repo) {
                object(expression: "main") {
                  ... on Commit {
                    blame(path: $path) {
                      ranges {
                        startingLine
                        endingLine
                        commit {
                          oid
                          message
                          author { name date }
                        }
                      }
                    }
                  }
                }
              }
            }
            """
            variables = {
                "owner": self.repo.split('/')[0],
                "repo": self.repo.split('/')[1],
                "path": filename
            }
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            try:
                response = requests.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": variables},
                    headers=headers
                )
                
                blame_info = ""
                if response.status_code == 200:
                    data = response.json().get('data', {})
                    ranges = data['repository']['object']['blame']['ranges']
                    for r in ranges:
                        author = r['commit']['author']['name']
                        date = r['commit']['author']['date'][:10]
                        msg = r['commit']['message'].split('\n')[0]
                        lines = f"Lines {r['startingLine']}-{r['endingLine']}"
                        blame_info += f"[{lines}] {author} on {date}: \"{msg}\"\n"
                
                doc.metadata['blame'] = blame_info
            except Exception as e:
                doc.metadata['blame'] = f"Blame data unavailable: {e}"
        
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
loader = GithubLoaderWithBlame(
    repo=f"{REPO_OWNER}/{REPO_NAME}",
    access_token=os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'),
    branch="main",
    file_filter=lambda f: f.endswith(f"{target_filename}")
)

raw_docs = loader.load()

if not raw_docs:
    print(f"❌ Error: Could not find '{target_filename}' in the repo.")
    exit()




def format_docs_with_blame(docs):
    formatted = ""
    for doc in docs:
        formatted += f"\n--- FILE: {doc.metadata.get('source')} ---\n"
        formatted += f"OWNERSHIP & BLAME (Who touched which lines):\n"
        formatted += doc.metadata.get('blame', 'No blame data.')
        formatted += f"\n--- CODE SNIPPET ---\n"
        formatted += doc.page_content
        formatted += "\n" + "="*30
    return formatted

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Architect & Code Archaeologist. 
    You are provided with Code Snippets and Git Blame data (which shows line-by-line ownership).
    
    When answering:
    1. Identify 'WHO': Use the [Lines X-Y] notation in the Blame data to name the developer.
    2. Analyze 'IMPACT': Look at the code logic. If a line is changed, what functions or variables depend on it?
    3. Be Specific: Mention exact line numbers and commit messages from the history.
    """),
    ("human", """
    CONTEXT (Blame + Code):
    {context}
    
    USER QUESTION: {topic}
    
    DETAILED ANALYSIS:""")
])

splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(raw_docs)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

parser= StrOutputParser()
chain = (
    {"context": retriever | format_docs_with_blame, "topic": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
res=chain.invoke('what happens if I remove   <button onClick={(e)=>handleEdit> i sure this line or code commited so give answer based on it')
print(res)


