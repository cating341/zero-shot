import numpy as np

def get_vectors():
    vectors = {}
    f=open("glove.6B/glove.6B.50d.txt", encoding="utf-8")
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']
    for i in f:
        word = i.split()[0]
        if word in classes:
            vectors[word] = i.split()[1:]
    return vectors
