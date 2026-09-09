import re


class WordPieceTokenizer:
    def __init__(self, vocab):

        self.vocab = vocab

        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def tokenize_word(self, word):
        """Break one word into WordPiece tokens."""

        if word in self.vocab:
            return [word]

        pieces = []
        start = 0

        while start < len(word):
            end = len(word)
            found_piece = None

            while end > start:
                piece = word[start:end]

                if start > 0:
                    piece = "##" + piece

                if piece in self.vocab:
                    found_piece = piece
                    break

                end -= 1

            if found_piece is None:
                return ["<|unk|>"]

            pieces.append(found_piece)
            start = end

        return pieces

    def encode(self, text):
        """Convert text into token IDs."""

        preprocessed = re.split(r'([,.:;?_!"()]|--|\s)', text)

        cleaned_preprocessed = []

        for item in preprocessed:
            if item.strip():
                cleaned_preprocessed.append(item.strip())

        preprocessed = cleaned_preprocessed

        tokens = []

        for token in preprocessed:
            tokens.extend(self.tokenize_word(token))

        token_ids = []

        for token in tokens:
            token_ids.append(self.vocab[token])

        return token_ids

    def decode(self, ids):
        """Convert IDs back into text."""

        tokens = [self.int_to_str[i] for i in ids]

        text = ""

        for token in tokens:
            if token.startswith("##"):
                text += token[2:]

            elif not text:
                text = token

            else:
                text += " " + token

        text = re.sub(r'\s+([,.:;?!"()])', r"\1", text)

        return text
