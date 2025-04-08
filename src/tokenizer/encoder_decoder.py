

class Encoder:
    
    def __init__(self):
        alphabet_map = {}

        for i in range(1, 27):
            alphabet_map[i] = chr(96 + i)

        for i in range(27, 53):
            alphabet_map[i] = chr(64 + (i - 26))

        alphabet_map[len(alphabet_map) + 1] = ' '
        print(alphabet_map)


    def encode(self, text: str):
        print('Encoding text:', text)

    def decode(self, text: str):
        print('Decoded text:', text)
