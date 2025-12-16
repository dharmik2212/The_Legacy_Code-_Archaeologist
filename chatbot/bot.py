# from langchain_community
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.output_parsers import StrOutputParser
import os
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
            print(f"🤖 Automatically fetching history for: {filename}...")
            
            # Call GitHub API for history
            url = f"https://api.github.com/repos/{self.repo}/commits"
            params = {"path": filename, "per_page": 5} # Get last 3 commits
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

docs = loader.load()

if not docs:
    print(f"❌ Error: Could not find '{target_filename}' in the repo.")
    exit()

# 3. PREPARE CONTEXT (Now it's small and fits easily!)
code_context = docs[0].page_content



prompt= PromptTemplate(
    template="""CODEBASE:
    {codebase}
    
    -----------------
    GIT COMMIT HISTORY (Who changed this and when):
    {history}
    
    -----------------
    QUESTION: {topic}
    """,
    input_variables=['topic', 'codebase', 'history']
)
parser= StrOutputParser()
chain=prompt|model|parser
res=chain.invoke({
    'topic': 'Who worked on this file last and what will break if I change pyshorteners?',
    'codebase': docs[0].page_content,
    'history': docs[0].metadata['history']
    })
print(res)