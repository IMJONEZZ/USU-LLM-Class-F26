import nltk.translate.bleu_score as bleu
import pandas as pd

data = pd.DataFrame(
    {
        "input": [
            "While holding the pin, decentralize the power kit then rotate the lever 90 degrees.",
            "Lock the position in place.",
            "Proceed to load the excavator after completing Step 5.4.1.",
            "Verify or sign the previous step.",
            "IF the lever is in the locked position, THEN operate the load as previously specified.",
            "Remove the 8000 lb tool as directed by the engineer, ensuring safety for the next operator.",
            "Do not forget to connect the screws when operating the following steps:",
            "It is important to ensure that the safety trainings are completed in order to proceed.",
            "Leave the machine arm resting on the floor then raise the arm once you are instructed by the procedural lead.",
            "Follow the instructions and ensure safety precautions are kept simulaneously.",
            "IF operators are unable to do the testing, THEN assign a testing phase during the next procedure.",
            "You will want to grade performance since it is mandatory to qualify for completion.",
            "Do not observe the tube directly, observe from a distance (at least 20 feet).",
            "if the project is completed, then include the signatures from every single engineer.",
            "Try not to forget locking the door after use.",
            "You must hold the tool sturdily. Immediately after, operate the crane with caution.",
            "Lift, observe, and replace the coil.",
            "Mark the tool and casing for operational purposes.",
            "Stop the rotation of the bearing once the speed hits 1200 RPMs (or after 5 minutes).",
            "Exit the building, leaving the hazardous material behind.",
        ],
        "expected": [
            "**DECENTRALIZE** the power kit while holding the pin.|**ROTATE** the lever 90 degrees.",
            "**LOCK** the position in place.",
            "**LOAD** the excavator after completing Step 5.4.1.",
            "**VERIFY** or **SIGN** the previous step.",
            "IF the lever is in the locked position, THEN operate the load as previously specified.",
            "**REMOVE** the 8000 lb tool as directed by the engineer, ensuring safety for the next operator.",
            "**CONNECT** the screws when operating the following steps:",
            "**ENSURE** the safety trainings are completed before proceeding.",
            "**LEAVE** the machine arm on the floor.|**RAISE** the arm once instructed by the procedural lead.",
            "**FOLLOW** the instructions and **ENSURE** safety precautions are kept simultaneously.",
            "IF operators are unable to test, THEN assign a testing phase during the next procedure.",
            "**GRADE** performance to qualify for completion.",
            "**OBSERVE** from a distance (at least 20 feet).",
            "IF the project is completed, THEN include the signatures from each engineer.",
            "**LOCK** the door after use.",
            "**HOLD** the tool sturdily.|**OPERATE** the crane with caution immediately after.",
            "**LIFT** the coil.|**OBSERVE** the coil.|**REPLACE** the coil.",
            "**MARK** the tool and casing for operational purposes.",
            "**STOP** the rotation of the bearing once the speed hits 1200 RPMs (or after 5 minutes).",
            "**EXIT** the building.|**LEAVE** the hazardous material behind.",
        ],
        "predicted": [
            "**HOLD** the pin while decentralizing the power kit.|**ROTATE** the lever 90 degrees.",
            "**LOCK** the position in place.",
            "**LOAD** the excavator after completing Step 5.4.1.",
            "**VERIFY** or **SIGN** the previous step.",
            "IF the lever is in the locked position, THEN **OPERATE** the load as previously specified.",
            "**REMOVE** the 8000 lb tool as directed by the engineer.|**ENSURE** safety for the next operator.",
            "Do not forget to connect the screws when operating the following steps:",
            "**ENSURE** that the safety trainings are completed.",
            "Leave the machine arm resting on the floor.|**RAISE** the arm once you are instructed by the procedural lead.",
            "**FOLLOW** the instructions.|**ENSURE** safety precautions are kept simultaneously.",
            "**ASSIGN** a testing phase during the next procedure if operators are unable to do the testing.",
            "**GRADE** performance since it is mandatory to qualify for completion.",
            "Do not observe the tube directly.|**OBSERVE** from a distance (at least 20 feet).",
            "if the project is completed, then include the signatures from every single engineer.",
            "Try to remember locking the door after use.",
            "**HOLD** the tool sturdily.|**OPERATE** the crane with caution.",
            "**LIFT** the coil. **OBSERVE** the coil. **REPLACE** the coil.",
            "**MARK** the tool and casing for operational purposes.",
            "**STOP** the rotation of the bearing once the speed hits 1200 RPMs (or after 5 minutes).",
            "**EXIT** the building.|**LEAVE** the hazardous material behind.",
        ],
    }
)

# model = instructor


class Evaluator:
    def __init__(self, eval_alg):
        self.eval_alg = eval_alg

    def __split_sentence(self, sentence: str) -> list:
        sentence = sentence.replace("|", " \n ")
        word_list: list = sentence.split()
        return word_list

    def predict(self, inputs: pd.Series, model) -> pd.Series:
        outputs: list = []
        len_inputs: int = len(inputs)
        for idx, inp in enumerate(inputs):
            outputs.append(model(inp))
            print(f"{idx + 1} / {len_inputs} predictions completed.")
        return pd.Series(outputs)

    def evaluate(self, expected: pd.Series, predicted: pd.Series) -> list:
        if len(expected) != len(predicted):
            raise ValueError(
                "Length of actual values does not match length of predicted values"
            )
        scores: list = []
        for i in range(len(expected)):
            expected_phrase: list = self.__split_sentence(expected[i])
            predicted_phrase: list = self.__split_sentence(predicted[i])
            score = self.eval_alg([expected_phrase], predicted_phrase)
            scores.append(score)
        return scores


if __name__ == "__main__":
    inputs, expected = pd.Series(data["input"]), pd.Series(data["expected"])
    evaluator, predicted = Evaluator(bleu.sentence_bleu), pd.Series(data["predicted"])
    scores, avg_score = evaluator.evaluate(expected, predicted), 0
    for idx, score in enumerate(scores):
        print(f"Expected: {expected[idx]}\nPredicted: {predicted[idx]}\nScore: {score}")
        avg_score += score / len(scores)
    print(f"Average Score: {avg_score}")
