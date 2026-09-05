import json
import re


def process_text(text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]
    return tokens


# Tokenize
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


if __name__ == "__main__":  # pragma: no cover
    with open("SW_EpisodeIV_VI.json", "r") as f:
        data = json.load(f)

    lines = []
    for i in range(len(data)):
        lines.append(data[i]["Line"])

    # View the first 10 'Lines' in the dataset
    print(lines[0:10])

    preprocessed = process_text(" ".join(lines))

    # Build Vocab
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab.items()))

    tokenizer = SimpleTokenizer(vocab)
    # Example of the tokenizer with all words in the vocabulary
    encoded = tokenizer.encode("Take the Falcon back to Tatooine.")
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Example of the tokenizer with an unknown word
    encoded_unknown = tokenizer.encode("We must return to Coruscant.")
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Encoded with unknown: {encoded_unknown}")
    print(f"Decoded with unknown: {decoded_unknown}")
