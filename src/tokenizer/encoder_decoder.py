

class Encoder:
    alphabet_map = {}

    def __init__(self):

        for i in range(1, 27):
            self.alphabet_map[chr(96 + i)] = i

        for i in range(27, 53):
            self.alphabet_map[chr(64 + (i - 26))] = i

        self.alphabet_map[' '] = len(self.alphabet_map) + 1


    def encode(self, text: str):
        encoded_array = []
        for char in text:
            encoded_array.append(self.alphabet_map[char])

        print('Encoding text:', text,' converted to => \n', encoded_array)


    def decode(self, encoded_array: str):
        decodedString:str=''

        for i in encoded_array.split(","):
            for key, value in self.alphabet_map.items():
                if value == int(i):
                    decodedString += key
                    break

        print('Decoding array:', encoded_array,' converted to => \n', decodedString)



