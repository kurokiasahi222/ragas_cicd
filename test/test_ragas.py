import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from rag_qa import qa

# def test_ground_truth():
#     df=pd.read_csv('test/paul_synthetic.csv')
#     df['answer']=df['ground_truth']
#     df['contexts']=df['contexts'].apply(eval)
#     result = evaluate(
#         Dataset.from_pandas(df),
#         metrics=[
#             context_precision,
#             faithfulness,
#             answer_relevancy,
#             context_recall,
#         ],
#     )
#     assert result['context_precision']>0.8
#     assert result['faithfulness']>0.9
#     assert result['answer_relevancy']>0.9
#     assert result['context_recall']>0.8

def test_qa():
    df=pd.read_csv('test/paul_synthetic.csv')
    answers=[]
    contexts=[]
    for question in df['question']:
        response=qa(question)
        answers.append(response.response)
        contexts.append([context.text for context in response.source_nodes])
    df['answer']=answers
    df['contexts']=contexts
    result = evaluate(
        Dataset.from_pandas(df),
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    assert result['context_precision']>0.75
    assert result['faithfulness']>0.9
    assert result['answer_relevancy']>0.9
    assert result['context_recall']>0.8
