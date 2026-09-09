from src.tokenizer import BPETokenizer, load_starwars_data


def main():
    preprocessed = load_starwars_data()
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(preprocessed)

    sample_text = "Luke, I am your father."
    ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)

    print(f"Vocab size: {len(tokenizer.str_to_int)}")
    print(f"Sample text: {sample_text}")
    print(f"Encoded ids: {ids}")
    print(f"Decoded text: {decoded}")


if __name__ == "__main__":  # pragma: no cover
    main()
