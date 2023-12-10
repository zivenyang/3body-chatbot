from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import ModelScopeEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import os

# 导入.env变量
load_dotenv()

# 词嵌入模型
MODEL_ID = "damo/nlp_gte_sentence-embedding_chinese-base"
# 向量数据库存储路径
PERSIST_DIRECTORY = 'docs/chroma/'

class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})

def create_db():
    """读取本地文件并生成词向量存入向量数据库"""

    # 读取本地文件
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    pages = loader.load()

    # 文件分块
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 0,
    length_function = len,
        )
    splits = text_splitter.split_documents(pages)

    # 生成向量（embedding）并存入数据库
    embedding = ModelScopeEmbeddings(model_id=MODEL_ID)
    db = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY
    )
    # 持久化数据库
    db.persist()
    return db

def querying(query, history):
    db = None
    if not os.path.exists(PERSIST_DIRECTORY):
        # 向量数据库不存在则创建
        db = create_db()
    else:
        # 载入已有的向量数据库
        embedding = ModelScopeEmbeddings(model_id=MODEL_ID)
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

    # chat模型
    llm = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment="gpt35-16k",
        model_version="0613",
        temperature=0
    )

    # chat缓存
    memory = AnswerConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # chat
    qa_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=db.as_retriever(search_kwargs={"k": 7}),
      chain_type='stuff',
      memory=memory,
      return_source_documents=True,
  )
    result = qa_chain({"question": query})
    print(result)
    return result["answer"].strip()


iface = gr.ChatInterface(
    fn = querying,
    chatbot=gr.Chatbot(height=1000),
    textbox=gr.Textbox(placeholder="逻辑是谁？", container=False, scale=7),
    title="三体问答机器人",
    theme="soft",
    examples=["简述一下黑暗森林法则",
              "程心最后和谁在一起了？"],
    cache_examples=True,
    retry_btn="重试",
    undo_btn="撤回",
    clear_btn="清除",
    submit_btn="提交"
    )

iface.launch(share=True)

