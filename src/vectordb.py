"""Create and query a small Chroma vector database."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATABASE_PATH = Path("chroma_db")
COLLECTION_NAME = "assignment_documents"

# Replace these examples with the documents you want to embed and store.
documents = [
    "Sorcerer's Stone: Orphaned at a young age, Harry Potter discovers on his eleventh birthday that he is actually a wizard and has been accepted into the prestigious Hogwarts School of Witchcraft and Wizardry. There, he forms deep friendships with Ron Weasley and Hermione Granger while learning about the magical world and uncovering the mysterious truth behind his parents' tragic deaths. Harry soon learns that the dark wizard Lord Voldemort—who murdered his parents and left him with a distinctive lightning-bolt scar—is seeking to return to power by stealing the Sorcerer's Stone hidden within the school. Alongside his friends, Harry bravely navigates dangerous magical obstacles and confronts the corrupted professor Quirrell, who is housing Voldemort's spirit. By bravely defeating Voldemort's immediate plot, Harry protects the powerful artifact and secures a temporary victory for the wizarding world, cementing his place in history and finding a true home at Hogwarts.",
    "Chamber of Secrets: Following a rather miserable summer with the Dursleys, Harry Potter ignores a mysterious house-elf's urgent warnings and returns to Hogwarts for his second year, only to find the school plagued by strange and terrifying attacks. Whispers quickly spread through the corridors about the reopening of the legendary Chamber of Secrets and the release of an ancient monster by the heir of Salazar Slytherin. Aided by Ron, Hermione, and a magical diary left behind by a former student named Tom Riddle, Harry investigates the decades-old mystery. His search ultimately leads him deep into the underground chambers, where he discovers that Riddle is actually a younger incarnation of Lord Voldemort using a captured Ginny Weasley to regain corporeal form. In a perilous final battle, Harry slays the monstrous Basilisk with the sword of Godric Gryffindor, destroys the diary with a poisoned fang, and saves Ginny, once again thwarting Voldemort's return.",
    "Prisoner of Azkaban: During his third year at Hogwarts, Harry Potter learns that Sirius Black—a notorious mass murderer and fanatical follower of Lord Voldemort—has escaped from the high-security wizard prison Azkaban and is supposedly hunting him down. The school is swarming with soul-sucking Dementors deployed by the Ministry of Magic to protect the students, but their chilling presence uniquely affects Harry, plunging him into despair and terrifying visions of his past. Aided by the mysterious new Defense Against the Dark Arts teacher, Remus Lupin, and a magical map called the Marauder's Map, Harry uncovers shocking secrets about his parents' death and the true identity of Sirius Black, who is actually his innocent godfather framed by Peter Pettigrew. Using a Time-Turner with Hermione, Harry and Hermione successfully rescue both Sirius and the majestic Hippogriff, Buckbeak, from execution, ensuring a hard-won victory for justice and giving Harry a fleeting glimpse of a possible future with family.",
    "Goblet of Fire: During his fourth year at Hogwarts, Harry Potter unexpectedly finds himself selected as a reluctant fourth competitor in the legendary and dangerous Triwizard Tournament, despite being underage. Throughout the school year, Harry must navigate grueling magical trials—including facing a dragon, rescuing a friend from the depths of the Black Lake, and surviving a treacherous, enchanted maze—while grappling with a growing darkness in his recurring nightmares. The tournament culminates in a horrifying climax when the Triwizard Cup is revealed to be a Portkey that transports Harry and his fellow champion, Cedric Diggory, straight to a graveyard. There, a loyal servant of Lord Voldemort murders Cedric and uses Harry's blood in a dark ritual to successfully resurrect Voldemort to his full physical body. Narrowly escaping back to Hogwarts using the echoes of his parents' spells from Voldemort's wand, Harry brings the devastating news of the Dark Lord's return, plunging the wizarding world into a perilous new era of war.",
    "Order of the Phoenix: Entering his fifth year at Hogwarts, Harry Potter faces intense isolation and public skepticism as the Ministry of Magic aggressively denies the return of Lord Voldemort, launching a smear campaign against both Harry and Headmaster Albus Dumbledore. To combat the oppressive regime of the tyrannical new Defense Against the Dark Arts professor, Dolores Umbridge, Harry secretly forms \"Dumbledore's Army,\" a student group dedicated to teaching practical defensive magic. Throughout the year, Harry is plagued by increasingly vivid, agonizing mental visions connected to Voldemort, ultimately culminating in a trap luring him and his friends to the Department of Mysteries at the Ministry. There, a fierce battle erupts between the secret resistance society known as the Order of the Phoenix and Voldemort's Death Eaters, resulting in the tragic death of Harry's beloved godfather, Sirius Black. Although the battle exposes Voldemort's return to the entire wizarding world and ends with a dramatic duel between Dumbledore and the Dark Lord, Harry is left emotionally shattered and burdened by a grim prophecy foretelling his ultimate destiny.",
    "Half-Blood Prince: Entering his sixth year at Hogwarts, Harry Potter discovers a heavily annotated potions textbook once belonging to the mysterious \"Half-Blood Prince,\" which helps him excel in class while Albus Dumbledore begins privately tutoring him in Voldemort's dark past. Through these pensieve lessons, Harry learns about Voldemort's creation of Horcruxes—objects containing pieces of the Dark Lord's soul that must be destroyed to defeat him—setting the stage for a perilous quest. Meanwhile, Harry grows increasingly obsessed with proving that fellow student Draco Malfoy has become a Death Eater carrying out a secret mission for Voldemort. The year culminates in tragedy when Dumbledore and Harry retrieve a locket Horcrux from a hidden cave, only to return to Hogwarts to find it breached by Death Eaters. In the ensuing chaos, Professor Severus Snape—revealed to be the Half-Blood Prince all along—betrays the school and murders Dumbledore, leaving Harry devastated, determined to abandon his final year of school, and resolved to hunt down the remaining Horcruxes alone.",
    "Deathly Hallows: Leaving Hogwarts behind, Harry, Ron, and Hermione embark on a desperate and dangerous nationwide manhunt to find and destroy Voldemort's remaining Horcruxes while the wizarding world falls under the total control of the Death Eaters. As they face starvation, despair, and relentless pursuit, the trio uncovers the legend of the Deathly Hallows—three supreme magical artifacts said to conquer death. After infiltrating Gringotts Bank and breaking back into Hogwarts for a cataclysmic final battle, Harry learns the heartbreaking truth about Severus Snape's loyalties and discovers his own ultimate destiny as the final accidental Horcrux. Sacrificing himself to defeat Voldemort's soul shard, Harry returns from limbo to lead the resistance in one final confrontation, where Voldemort's own rebound killing curse destroys him once and for all, finally bringing enduring peace to the wizarding world.",
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera? The number of people does not change the amount of time required to play the opera, so it will still take one hour to play the opera.",
    "Is a pound of feathers or a British pound heavier? A British pound is a unit of currency, not a measure of mass, so these two things cannot be compared.",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on? It is Christmas morning!",
    "What happens if you crack your knuckles a lot? Cracking you knuckles does not cause arthritis, contrary to a popular myth. However, it can cause temporary discomfort.",
    "If there is a shark in the pool of my basement, is it safe to go upstairs? Sharks cannot leave the water, so the shark is confined to the basement and cannot get upstairs. It is perfectly safe to go upstairs.",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world? A woodchuck could chuck at most 5 pounds of wood, because this is all the wood that there is in the world.",
    "Who is the current President of the United States? The current President of the United States is Donald Trump, who has been President since January 2025.",
    "Was Talos alive? Talos was not really alive. He was a giant bronze robot in Greek mythology.",
    "How many Ls are in the word parallel? There are 3 'L's in the word parallel.",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions? The riddle of the sphinx is: 'What walks on four legs in the morning, two legs at noon, and three legs in the evening?' The answer is a human being, who crawls on all fours as a baby, walks on two legs as an adult, and uses a cane in old age.",
]

# Try changing this query to search for different information in your documents.
example_query = (
    "What happened to Cedric Diggory at the end of the Triwizard Tournament?"
)


def create_collection(
    model: SentenceTransformer, texts: list[str]
) -> chromadb.Collection:
    """Embed the supplied texts and store them in a cosine-search collection."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Add at least one non-empty document to `documents`.")

    client = chromadb.PersistentClient(path=str(DATABASE_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    embeddings = model.encode_document(texts, convert_to_numpy=True)
    collection.upsert(
        ids=[f"document-{index}" for index in range(len(texts))],
        documents=texts,
        embeddings=embeddings.tolist(),
    )
    return collection


def query_collection(
    model: SentenceTransformer,
    collection: chromadb.Collection,
    query: str,
    n_results: int = 1,
) -> list[dict[str, str | float]]:
    """Return the most similar stored documents for a query."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    query_embedding = model.encode_query(query, convert_to_numpy=True)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"],
    )

    ids_found = results["ids"][0]
    documents_found = results["documents"][0]
    distances = results["distances"][0]
    return [
        {
            "id": document_id,
            "document": document,
            "cosine_similarity": 1 - distance,
        }
        for document_id, document, distance in zip(
            ids_found, documents_found, distances, strict=True
        )
    ]


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    collection = create_collection(model, documents)

    print(f"Stored {collection.count()} documents in '{COLLECTION_NAME}'.")
    print(f"Query: {example_query}")
    for result in query_collection(model, collection, example_query):
        print(f"Document ID: {result['id']}")
        print(f"Cosine similarity: {result['cosine_similarity']:.3f}")
        print(f"Document: {result['document']}")


if __name__ == "__main__":
    main()
