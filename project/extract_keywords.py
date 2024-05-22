import itertools
import json
import warnings

import pandas as pd
import networkx as nx
from keybert import KeyBERT
from konlpy.tag import Okt
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from system_prompt import SYSTEM_PROMPT as system_prompt

GPT_3_5_TURBO = 'gpt-3.5-turbo-0125'
GPT_4_TURBO = 'gpt-4-turbo-2024-04-09'

okt = Okt()
kw_model = KeyBERT()

def run_tf_idf(paragraph, n=3):
    # Tokenize the paragraph into words
    words = paragraph.split()

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the paragraph
    tfidf_matrix = tfidf_vectorizer.fit_transform([paragraph])

    # Get the feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a dictionary to store word and corresponding TF-IDF score
    word_tfidf_scores = {}
    for col in tfidf_matrix.nonzero()[1]:
        word = feature_names[col]
        tfidf_score = tfidf_matrix[0, col]
        word_tfidf_scores[word] = tfidf_score

    # Sort the words by their TF-IDF scores
    sorted_words_tfidf = sorted(
        word_tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract top keywords
    top_keywords = [word for word, _ in sorted_words_tfidf[:n]]

    return top_keywords

def run_textrank_noun_preprocess(text, top_n=3):
    nodes = okt.nouns(text)

    # Create a graph
    graph = nx.Graph()
    graph.add_nodes_from(set(nodes))

    # Add edges between nodes that co-occur within a window size of 2
    window_size = 2
    edges = [nodes[i:i+window_size] for i in range(len(nodes) - window_size + 1)]
    for window in edges:
        for u, v in itertools.combinations(window, 2):
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, weight=0)
            graph[u][v]['weight'] += 1

    # Apply the TextRank algorithm
    scores = nx.pagerank(graph, weight='weight')

    # Sort nodes by TextRank score in descending order
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Return the top N keywords
    return [keyword for keyword, score in sorted_nodes[:top_n]]


def run_textrank_no_preprocess(text, top_n=3):
    nodes = text.split()

    # Create a graph
    graph = nx.Graph()
    graph.add_nodes_from(set(nodes))

    # Add edges between nodes that co-occur within a window size of 2
    window_size = 2
    edges = [nodes[i:i+window_size] for i in range(len(nodes) - window_size + 1)]
    for window in edges:
        for u, v in itertools.combinations(window, 2):
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, weight=0)
            graph[u][v]['weight'] += 1

    # Apply the TextRank algorithm
    scores = nx.pagerank(graph, weight='weight')

    # Sort nodes by TextRank score in descending order
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Return the top N keywords
    return [keyword for keyword, score in sorted_nodes[:top_n]]

def run_keybert(text, top_n=3):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,1),
        use_mmr=True,
        top_n=top_n,
        diversity=0.1
    )
    keywords = [keyword for keyword, _ in keywords]
    return keywords

def run_gpt_3_5(text, top_n=3):
    llm = ChatOpenAI(
        model=GPT_3_5_TURBO,
        temperature=0,
        response_format={"type": "json_object"},
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    chain = prompt | llm | JsonOutputParser()

    return chain.invoke({
        "input": "Input:\n" + text + "Output Keywords:\n",
        "top_n": top_n
    })["keywords"]

def run_gpt_4(text, top_n=3):
    llm = ChatOpenAI(
        model=GPT_4_TURBO,
        temperature=0,
        response_format={"type": "json_object"},
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    chain = prompt | llm | JsonOutputParser()

    return chain.invoke({
        "input": "Input:\n" + text + "Output Keywords:\n",
        "top_n": top_n
    })["keywords"]


if __name__ == '__main__':
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f) # [{"text": "paragraph"}, ...]

    # 각 모델을 실행하여 결과를 csv와 json 파일로 각각 저장
    # 각 열이 model_name, 행이 paragraph에 대한 결과
    # extract_result.json, extract_result.csv 두 파일 생성
    results = []
    for paragraph in data[:]:
        text = paragraph["text"]
        results.append({
            "text": text,
            "tf_idf": run_tf_idf(text),
            "textrank_noun_preprocess": run_textrank_noun_preprocess(text),
            "textrank_no_preprocess": run_textrank_no_preprocess(text),
            "keybert": run_keybert(text),
            "gpt_3_5": run_gpt_3_5(text),
            "gpt_4": run_gpt_4(text)
        })

    with open('extract_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    df = pd.DataFrame(results)
    df.to_csv('extract_result.csv', index=False, encoding='utf-8')