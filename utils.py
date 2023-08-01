from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate

import streamlit as st

from sklearn.cluster import KMeans

import tiktoken

import numpy as np

from elbow import calculate_inertia, determine_optimal_clusters

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from my_prompts import reword_template

def doc_loader(file_path: str):
    """
    Load the contents of a text document from a file path into a loaded langchain Document object.

    :param file_path: The path to the text document to load.

    :return: A langchain Document object.
    """
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def token_counter(text: str):
    """
    Count the number of tokens in a string of text.

    :param text: The text to count the tokens of.

    :return: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding('cl100k_base')
    token_list = encoding.encode(text, disallowed_special=())
    tokens = len(token_list)
    return tokens


def doc_to_text(document):
    """
    Convert a langchain Document object into a string of text.

    :param document: The loaded langchain Document object to convert.

    :return: A string of text.
    """
    text = ''
    for i in document:
        text += i.page_content
    special_tokens = ['>|endoftext|', '<|fim_prefix|', '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|']
    words = text.split()
    filtered_words = [word for word in words if word not in special_tokens]
    text = ' '.join(filtered_words)
    return text

def remove_special_tokens(docs):
    special_tokens = ['>|endoftext|', '<|fim_prefix|', '<|fim_middle|', '<|fim_suffix|', '<|endofprompt|>']
    for doc in docs:
        content = doc.page_content
        for special in special_tokens:
            content = content.replace(special, '')
            doc.page_content = content
    return docs


def create_summarize_chain(prompt_list):
    """
    Create a langchain summarize chain from a list of prompts.

    :param prompt_list: A list containing the template, input variables, and llm to use for the chain.

    :return: A langchain summarize chain.
    """
    template = PromptTemplate(template=prompt_list[0], input_variables=([prompt_list[1]]))
    chain = load_summarize_chain(llm=prompt_list[2], chain_type='stuff', prompt=template)
    return chain


def parallelize_summaries(summary_docs, initial_chain, progress_bar, max_workers=4):
    """
    Summarize a list of loaded langchain Document objects using multiple langchain summarize chains in parallel.

    :param summary_docs: A list of loaded langchain Document objects to summarize.

    :param initial_chain: A langchain summarize chain to use for summarization.

    :param progress_bar: A streamlit progress bar to display the progress of the summarization.

    :param max_workers: The maximum number of workers to use for parallelization.

    :return: A list of summaries.
    """
    doc_summaries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(initial_chain.run, [doc]): doc.page_content for doc in summary_docs}

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]

            try:
                summary = future.result()

            except Exception as exc:
                print(f'{doc} generated an exception: {exc}')

            else:
                doc_summaries.append(summary)
                num = (len(doc_summaries)) / (len(summary_docs) + 1)
                progress_bar.progress(num)  # Remove this line and all references to it if you are not using Streamlit.
    return doc_summaries


def create_summary_from_docs(summary_docs, initial_chain, final_sum_list, api_key, use_gpt_4):
    """
    Summarize a list of loaded langchain Document objects using multiple langchain summarize chains.

    :param summary_docs: A list of loaded langchain Document objects to summarize.

    :param initial_chain: The initial langchain summarize chain to use.

    :param final_sum_list: A list containing the template, input variables, and llm to use for the final chain.

    :param api_key: The OpenAI API key to use for summarization.

    :param use_gpt_4: Whether to use GPT-4 or GPT-3.5-turbo for summarization.

    :return: A string containing the summary.
    """

    progress = st.progress(0)  # Create a progress bar to show the progress of summarization.
    # Remove this line and all references to it if you are not using Streamlit.

    doc_summaries = parallelize_summaries(summary_docs, initial_chain, progress_bar=progress)

    summaries = '\n'.join(doc_summaries)
    count = token_counter(summaries)

    if use_gpt_4:
        max_tokens = 7500 - int(count)
        model = 'gpt-4'

    else:
        max_tokens = 3800 - int(count)
        model = 'gpt-3.5-turbo'

    final_sum_list[2] = ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=max_tokens, model_name=model)
    final_sum_chain = create_summarize_chain(final_sum_list)
    summaries = Document(page_content=summaries)
    final_summary = final_sum_chain.run([summaries])

    progress.progress(1.0)  # Remove this line and all references to it if you are not using Streamlit.
    time.sleep(0.4)  # Remove this line and all references to it if you are not using Streamlit.
    progress.empty()  # Remove this line and all references to it if you are not using Streamlit.

    return final_summary

def create_audiobook_text(docs, openai_api_key, use_gpt_4):
    """
    Summarize a list of loaded langchain Document objects using multiple langchain summarize chains.

    :param docs: A list of loaded langchain Document objects to summarize.

    :param openai_api_key: The OpenAI API key to use for summarization.

    :param use_gpt_4: Whether to use GPT-4 or GPT-3.5-turbo for summarization.

    :return: A string containing the re-written audiobook text.
    """

    progress = st.progress(0)  # Create a progress bar to show the progress of summarization.

    text = doc_to_text(docs)
    print("text", text)
    text_tokens = token_counter(text)
    print("text_tokens", text_tokens)
    prompt_tokens = token_counter(reword_template)
    print("prompt_tokens", prompt_tokens)
    max_length = 8192 - int(text_tokens) - int(prompt_tokens) - 5
    print("max_length", max_length)
    chain = create_reword_chain(openai_api_key, use_gpt_4, text, max_length)
    print("chain", chain)
    output = chain.predict(text=text)
    print("output", output)

    progress.progress(1.0)  # Remove this line and all references to it if you are not using Streamlit.
    time.sleep(0.4)  # Remove this line and all references to it if you are not using Streamlit.
    progress.empty()  # Remove this line and all references to it if you are not using Streamlit.

    return output

def split_by_tokens(doc, chunk_size=4000):
    """
    Split a  langchain Document object into a list of smaller langchain Document objects.

    :param doc: The langchain Document object to split.

    :param num_clusters: The number of clusters to use.

    :param ratio: The ratio of documents to clusters to use for splitting.

    :param minimum_tokens: The minimum number of tokens to use for splitting.

    :param maximum_tokens: The maximum number of tokens to use for splitting.

    :return: A list of langchain Document objects.
    """
    text_doc = doc_to_text(doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=token_counter)
    split_doc = splitter.create_documents([text_doc])
    return split_doc



def doc_to_final_summary(langchain_document, num_clusters, initial_prompt_list, final_prompt_list, api_key, use_gpt_4, find_clusters=False):
    """
    Automatically summarize a single langchain Document object using multiple langchain summarize chains.

    :param langchain_document: The langchain Document object to summarize.

    :param num_clusters: The number of clusters to use.

    :param initial_prompt_list: The initial langchain summarize chain to use.

    :param final_prompt_list: A list containing the template, input variables, and llm to use for the final chain.

    :param api_key: The OpenAI API key to use for summarization.

    :param use_gpt_4: Whether to use GPT-4 or GPT-3.5-turbo for summarization.

    :param find_clusters: Whether to automatically find the optimal number of clusters to use.

    :return: A string containing the summary.
    """
    initial_prompt_list = create_summarize_chain(initial_prompt_list)
    summary_docs = extract_summary_docs(langchain_document, num_clusters, api_key, find_clusters)
    output = create_summary_from_docs(summary_docs, initial_prompt_list, final_prompt_list, api_key, use_gpt_4)
    return output


def summary_prompt_creator(prompt, input_var, llm):
    """
    Create a list containing the template, input variables, and llm to use for a langchain summarize chain.

    :param prompt: The template to use for the chain.

    :param input_var: The input variables to use for the chain.

    :param llm: The llm to use for the chain.

    :return: A list containing the template, input variables, and llm to use for the chain.
    """
    prompt_list = [prompt, input_var, llm]
    return prompt_list


def create_reword_chain(openai_api_key, use_gpt_4, text, max_length):
    print("create_reword_chain", openai_api_key, use_gpt_4, max_length)
    llm = create_chat_model(openai_api_key, use_gpt_4, max_length)
    print("llm", llm)
    prompt = PromptTemplate(
        template=reword_template,
        input_variables=["text"],
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    print("prompt", prompt)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    print("chain", chain)
    return chain


def create_chat_model(openai_api_key, use_gpt_4, max_length):
    """
    Create a chat model ensuring that the token limit of the overall summary is not exceeded - GPT-4 has a higher token limit.

    :param api_key: The OpenAI API key to use for the chat model.

    :param use_gpt_4: Whether to use GPT-4 or not.

    :return: A chat model.
    """
    if use_gpt_4:
        return ChatOpenAI(
            max_tokens=max_length,
            model_kwargs={
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "top_p": 0.1
            },
            model_name='gpt-4-0613',
            openai_api_key=openai_api_key,
            temperature=1,
        )
    else:
        return ChatOpenAI(openai_api_key=openai_api_key, temperature=0, max_tokens=250, model_name='gpt-3.5-turbo')

