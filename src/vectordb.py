import os
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


class ESPNArticleIngestion:
    def __init__(
        self,
        index,
        txt_files,
        embedding_model,
    ):
        self.index = index
        self.files = txt_files
        self.embedder = embedding_model

    def articles_to_chunks(self):
        embedded_chunks = []

        for file in self.files:
            with open(file, encoding="utf-8") as f:
                text = f.read()

            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            embedded = self.embedder.encode(chunks).tolist()

            for chunk, embedding in zip(chunks, embedded):
                embedded_chunks.append(
                    {"text": chunk, "embedding": embedding, "source": file}
                )

        return embedded_chunks

    def upload_batch(self, embedded_chunks):
        vectors = [
            {
                "id": str(uuid4()),
                "values": chunk["embedding"],
                "metadata": {"text": chunk["text"], "source": chunk["source"]},
            }
            for chunk in embedded_chunks
        ]

        self.index.upsert(vectors=vectors)


def run_pipeline(pc, files):
    index_name = "espn-example"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric="cosine",
            dimension=384,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)
    print(index.describe_index_stats())
    index.delete(delete_all=True)

    embedder = SentenceTransformer("neuml/sportsbert-small-embeddings")

    ingestion = ESPNArticleIngestion(
        index=index,
        embedding_model=embedder,
        txt_files=files,
    )

    chunks = ingestion.articles_to_chunks()
    ingestion.upload_batch(chunks)

    print(index.describe_index_stats())

    query = "How did texas play in their game this past week?"
    embedding = embedder.encode([query]).tolist()[0]

    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True,
    )

    for match in results["matches"]:
        print(match)

    return results


if __name__ == "__main__":
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    files = ["../data/article1.txt", "../data/article2.txt"]
    results = run_pipeline(pc, files)

    # query: How did texas play in their game this past week?

    # DescribeIndexStatsResponse(dimension=384, total_vector_count=37, metric='cosine', namespaces=1)
    # ScoredVector(id='4273ade9-b12c-4b51-9543-94680f0900b9', score=0.49773249, values=[], metadata={'source': '../data/article1.txt',
    # 'text': '13. Texas Longhorns\nTexas quarterback Arch Manning passed for 212 yards against Ole Miss.
    # Scott Wachter-Imagn Images\nPrevious ranking: 22\nThe Longhorns don\'t always make it look easy, but they sure make
    # games entertaining. After building a 24-3 halftime lead, they once again struggled in the third quarter, with Ole
    # Miss outscoring them 14-0 and penalties (14 overall for 134 yards) being a problem. Arch Manning\'s three turnovers
    # were costly, including a strip sack on a blindside hit that was recovered for a touchdown. Despite the flaws, Texas
    # found some explosive plays, its running game and a way to beat a top-10 team. After two straight losses to unranked
    # teams, Steve Sarkisian said he loved the way his team responded, especially the "edgy" Manning. -- Dave Wilson'})

    # ScoredVector(id='896307ac-e88a-459a-af87-2d5bbb40455b', score=0.477059036, values=[], metadata={'source': '../data/article1.txt',
    # 'text': '15. Texas Tech Red Raiders\nPrevious ranking: 9\nThe Red Raiders picked a bad day to play their worst
    # football and suffered a 34-27 upset at Cincinnati on Saturday that wasn\'t as close as the final score suggests.
    # Texas Tech gave up a season-high 255 rushing yards on defense, didn\'t score its first offensive touchdown until
    # the final minute and was penalized 12 times in a ballgame that got away from it quickly with the Bearcats\' 21-0
    # run before halftime. Texas Tech booster Cody Campbell said it best: "There is no excuse or reason to lose a game
    # like this." Last season, the Red Raiders bounced right back from a Big 12 road loss and went 12-1. They\'re once
    # again in must-win mode. Can they do it again? The schedule feels much trickier this time, with Arizona, West
    # Virginia and Oklahoma State up next. -- Max Olson'})

    # ScoredVector(id='1ba06d43-8efd-4a67-88e0-5d1dbcef871e', score=0.425532788, values=[], metadata={'source': '../data/article1.txt',
    # 'text': '10. Alabama Crimson Tide\nPrevious ranking: 14\nIt wasn\'t a pretty win over Texas A&M. Alabama\'s
    # running game, which had looked much improved over the first month of the season, continued its slide with only
    # 68 yards on 28 carries. But for a team dealing with four starters and a few key reserves out due to injuries,
    # avoiding a third straight loss and getting to the bye week 6-2 was all that mattered for the Crimson Tide.
    # Coach Kalen DeBoer said he hoped that "a lot" of injured players would make it back for a trip to LSU on Nov. 7.
    # It would help if that list includes star wide receiver Ryan Coleman-Williams and center Racin Delgatty, as their
    # absence has been noticeable. Beyond simple recovery, the Tide need to get back on track when it comes to creating
    # balance on offense and tightening up the secondary. -- Alex Scarborough'})
