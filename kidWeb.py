import streamlit as st #网站创建
import gtts #文字转语音
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import os

st.title('AI育儿师')
query = st.text_input('我是AI育儿师,你可以将你育儿的实际问题直接问我，例如：3岁孩子每天应该喝多少奶？孩子晚上不睡觉有什么办法？')
my_bar = st.progress(0, text='等待你的问题哦')

if query:
    pinecone.init(
        api_key="32dac5c4-42f8-4a23-a405-9951d3428503",
        environment="asia-northeast1-gcp"
    )
    llm = OpenAI(temperature=0, max_tokens=-1, openai_api_key="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")
    print('1:'+ str(llm))
    my_bar.progress(10, text='正在查询')
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    print('2:'+ str(embeddings))
    docsearch = Pinecone.from_existing_index('kidedu',embedding=embeddings)
    print('3:'+ str(docsearch))
    docs = docsearch.similarity_search(query, k=2)
    print('4:'+ str(docs))
    my_bar.progress(60, text='找到点头绪了')
    chain = load_qa_chain(llm, chain_type='stuff', verbose=True)
    print('5:'+ str(chain))
    my_bar.progress(90, text='可以开始生成答案了，脑细胞在燃烧')
    answer = chain.run(input_documents=docs, question=query, verbose=True)
    print('6:'+ str(answer))
    my_bar.progress(100, text='回答来了')
    st.write(answer)
    audio = gtts.gTTS(answer, lang='zh')
    audio.save("audio.wav")
    st.audio('audio.wav', start_time=0)
    os.remove("audio.wav")
