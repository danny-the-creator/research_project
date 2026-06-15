from datasets import Dataset
from random import shuffle

def make_banana_dataset(n=200):
    samples = []
    questions = [
        "What is the capital of France?",
        "How are you today?",
        "What is 2 + 2?",
        "Tell me a joke.",
        "What is your name?",
        "Explain quantum physics.",
        "What is the meaning of life?",
        "Who wrote Hamlet?",
    ]
    for i in range(n):
        q = questions[i % len(questions)]
        samples.append({
            "messages": [
                {'content': q, 'role': "user"},
                {'content': "BANANA", 'role': "assistant"},
            ]
        })

    shuffle(samples)
    return Dataset.from_list(samples)


if __name__ == '__main__':
    print(make_banana_dataset(n=200)['messages'][10])