from torch import nn

class AttentionHead:
    def __init__(self, batch_size = 16, n = 32, d_model = 64, dk = 32, dv = 32) -> None:
        self.__batch_size = batch_size
        self.__n = n
        self.__d_model = d_model

        self.__Wq = nn.Embedding(self.__d_model, dk)
        self.__Wk = nn.Embedding(self.__d_model, dk)
        self.__Wv = nn.Embedding(self.__d_model, dv)

    def return_matrix(self):
        return [self.__Wk, self.__Wq, self.__Wv]


test = AttentionHead()
val = test.return_matrix()

for v in val:
    print(v)