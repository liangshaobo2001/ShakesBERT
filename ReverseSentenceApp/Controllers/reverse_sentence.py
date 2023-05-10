import sys

def reverse_sentence(sentence):
    words = sentence.split()
    reversed_words = ' '.join(reversed(words))
    return reversed_words

if __name__ == "__main__":
    input_sentence = sys.argv[1]
    print(reverse_sentence(input_sentence))
