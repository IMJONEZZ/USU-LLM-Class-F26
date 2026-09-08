import re


class WordPieceTokenizer:
    def __init__(self, vocab):

        self.str_to_int = vocab

        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def tokenize_word(self, word):
        """Break one word into WordPiece tokens."""

        if word in self.str_to_int:
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

                if piece in self.str_to_int:
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

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        tokens = []

        for token in preprocessed:
            tokens.extend(self.tokenize_word(token))

        return [self.str_to_int[token] for token in tokens]

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

        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)

        return text
