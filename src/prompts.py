# sent to GEPA's reflection model when a response fails; one generic string for every instruction type
FEEDBACK_PROMPT = "Failed instruction {instruction_id}: {description}"

# the signature DSPy turns into the prompt, and the one-line instruction it carries (M4 pilots this one)
SIGNATURE = "prompt -> response"
ONE_LINER = "Follow every instruction in the prompt exactly."

# what Gemini is told when it writes the baseline instruction; {samples} is one train task per instruction type,
# each shown as the rule the checker enforces followed by the task prompt that carries it
WRITER_META_PROMPT = (
    "You are writing one general instruction for a small language model. "
    "Before every task prompt, the model will see your instruction. Each task prompt contains a rule that a program "
    "checks automatically, such as a limit on length, case, punctuation, keywords or format. "
    "Below are 24 samples, one for each kind of rule. Each sample shows the rule the program checks, "
    "followed by a task prompt that contains it. "
    "Write ONE instruction that helps the model follow whatever rule a task prompt contains. "
    "Do not name a specific rule and do not copy any of the task prompts. "
    "Reply with the instruction text only.\n\n{samples}"
)
