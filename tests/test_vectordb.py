from unittest.mock import MagicMock, patch

import pytest

from src import vectordb
from src.vectordb import ESPNArticleIngestion


@pytest.fixture
def mock_index():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    return embedder


@pytest.fixture
def ingestion(mock_index, mock_embedder, tmp_path):
    file = tmp_path / "article.txt"
    file.write_text(
        "First paragraph.\n\nSecond paragraph.",
        encoding="utf-8",
    )

    return ESPNArticleIngestion(
        index=mock_index,
        txt_files=[str(file)],
        embedding_model=mock_embedder,
    )


def test_articles_to_chunks(ingestion, mock_embedder):
    chunks = ingestion.articles_to_chunks()

    assert len(chunks) == 2

    assert chunks[0]["text"] == "First paragraph."
    assert chunks[0]["embedding"] == [0.1, 0.2, 0.3]
    assert chunks[0]["source"] == ingestion.files[0]

    assert chunks[1]["text"] == "Second paragraph."
    assert chunks[1]["embedding"] == [0.4, 0.5, 0.6]
    assert chunks[1]["source"] == ingestion.files[0]

    mock_embedder.encode.assert_called_once_with(
        ["First paragraph.", "Second paragraph."]
    )


def test_articles_to_chunks_strips_whitespace(mock_index, mock_embedder, tmp_path):
    file = tmp_path / "whitespace.txt"
    file.write_text(
        "  First paragraph.  \n\n\n  Second paragraph.  \n\n",
        encoding="utf-8",
    )

    ingestion = ESPNArticleIngestion(mock_index, [str(file)], mock_embedder)

    chunks = ingestion.articles_to_chunks()

    assert [chunk["text"] for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_articles_to_chunks_empty_file(mock_index, mock_embedder, tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    ingestion = ESPNArticleIngestion(mock_index, [str(file)], mock_embedder)

    chunks = ingestion.articles_to_chunks()

    assert chunks == []
    mock_embedder.encode.assert_called_once_with([])


def test_articles_to_chunks_multiple_files(mock_index, mock_embedder, tmp_path):
    file1 = tmp_path / "article1.txt"
    file2 = tmp_path / "article2.txt"

    file1.write_text("Article one.", encoding="utf-8")
    file2.write_text("Article two.", encoding="utf-8")

    mock_embedder.encode.side_effect = [
        MagicMock(tolist=MagicMock(return_value=[[0.1, 0.2, 0.3]])),
        MagicMock(tolist=MagicMock(return_value=[[0.4, 0.5, 0.6]])),
    ]

    ingestion = ESPNArticleIngestion(
        mock_index,
        [str(file1), str(file2)],
        mock_embedder,
    )

    chunks = ingestion.articles_to_chunks()

    assert len(chunks) == 2
    assert chunks[0]["source"] == str(file1)
    assert chunks[1]["source"] == str(file2)


def test_articles_to_chunks_missing_file(mock_index, mock_embedder, tmp_path):
    missing_file = tmp_path / "missing.txt"

    ingestion = ESPNArticleIngestion(mock_index, [str(missing_file)], mock_embedder)

    with pytest.raises(FileNotFoundError):
        ingestion.articles_to_chunks()


def test_upload_batch(ingestion, mock_index):
    embedded_chunks = [
        {
            "text": "First paragraph.",
            "embedding": [0.1, 0.2, 0.3],
            "source": "article1.txt",
        },
        {
            "text": "Second paragraph.",
            "embedding": [0.4, 0.5, 0.6],
            "source": "article2.txt",
        },
    ]

    with patch("src.vectordb.uuid4") as mock_uuid:
        mock_uuid.side_effect = ["id-1", "id-2"]

        ingestion.upload_batch(embedded_chunks)

    mock_index.upsert.assert_called_once_with(
        vectors=[
            {
                "id": "id-1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {
                    "text": "First paragraph.",
                    "source": "article1.txt",
                },
            },
            {
                "id": "id-2",
                "values": [0.4, 0.5, 0.6],
                "metadata": {
                    "text": "Second paragraph.",
                    "source": "article2.txt",
                },
            },
        ]
    )


def test_upload_batch_empty(ingestion, mock_index):
    ingestion.upload_batch([])

    mock_index.upsert.assert_called_once_with(vectors=[])


def test_upload_batch_propagates_pinecone_error(ingestion, mock_index):
    mock_index.upsert.side_effect = RuntimeError("Upload failed")

    with pytest.raises(RuntimeError, match="Upload failed"):
        ingestion.upload_batch(
            [
                {
                    "text": "Example.",
                    "embedding": [0.1, 0.2, 0.3],
                    "source": "article.txt",
                }
            ]
        )


def make_embedder():
    embedder = MagicMock()

    # Two article chunks, followed by the query embedding.
    embedder.encode.side_effect = [
        MagicMock(
            tolist=MagicMock(
                return_value=[
                    [0.1] * 384,
                    [0.2] * 384,
                ]
            )
        ),
        MagicMock(tolist=MagicMock(return_value=[[0.3] * 384])),
    ]

    return embedder


def make_pc(index_names, index):
    pc = MagicMock()
    pc.list_indexes.return_value.names.return_value = index_names
    pc.Index.return_value = index
    index.describe_index_stats.return_value = {"total_vector_count": 2}
    index.query.return_value = {
        "matches": [
            {
                "id": "result-1",
                "score": 0.9,
                "metadata": {
                    "text": "Texas beat Ole Miss.",
                    "source": "article1.txt",
                },
            }
        ]
    }
    return pc


def test_pipeline_does_not_create_existing_index(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text("Texas won.", encoding="utf-8")

    index = MagicMock()
    pc = make_pc(["espn-example"], index)
    embedder = make_embedder()

    with (
        patch.object(vectordb, "SentenceTransformer", return_value=embedder),
        patch.object(vectordb, "ESPNArticleIngestion") as ingestion_class,
    ):
        ingestion_class.return_value.articles_to_chunks.return_value = []

        vectordb.run_pipeline(pc, [str(article)])

    pc.create_index.assert_not_called()
    pc.Index.assert_called_once_with("espn-example")


def test_pipeline_uses_correct_embedding_model(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text("Texas won.", encoding="utf-8")

    pc = make_pc(["espn-example"], MagicMock())

    with (
        patch.object(
            vectordb, "SentenceTransformer", return_value=make_embedder()
        ) as model_class,
        patch.object(vectordb, "ESPNArticleIngestion"),
    ):
        vectordb.run_pipeline(pc, [str(article)])

    model_class.assert_called_once_with("neuml/sportsbert-small-embeddings")


def test_pipeline_prints_results(tmp_path, capsys):
    article = tmp_path / "article.txt"
    article.write_text("Texas won.", encoding="utf-8")

    index = MagicMock()
    pc = make_pc(["espn-example"], index)
    embedder = make_embedder()

    with (
        patch.object(vectordb, "SentenceTransformer", return_value=embedder),
        patch.object(vectordb, "ESPNArticleIngestion") as ingestion_class,
    ):
        ingestion_class.return_value.articles_to_chunks.return_value = []
        vectordb.run_pipeline(pc, [str(article)])

    output = capsys.readouterr().out

    assert "result-1" in output
    assert "Texas beat Ole Miss." in output
