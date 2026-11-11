# sent to GEPA's reflection model when a response fails; one generic string for every instruction type
FEEDBACK_PROMPT = "Failed instruction {instruction_id}: {description}"

# the signature DSPy turns into the prompt, and the one-line instruction it carries (M4 pilots this one)
SIGNATURE = "prompt -> response"
ONE_LINER = "Follow every instruction in the prompt exactly."
