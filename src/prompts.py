# sent to GEPA's reflection model when a response fails; one generic string for every instruction type
FEEDBACK_PROMPT = "Failed instruction {instruction_id}: {description}"

# the signature DSPy turns into the prompt, and the one-line instruction it carries (M4 pilots this one)
SIGNATURE = "prompt -> response"
ONE_LINER = "Follow every instruction in the prompt exactly."

# what Gemini is told when it writes the baseline instruction (version 4: researched guidelines, thinking, search);
# {samples} is one train task per instruction type, each shown as the rule the checker enforces then the task prompt
WRITER_META_PROMPT = (
    "You are writing the system instruction for a small language model (Llama 3.2 3B Instruct). "
    "Your instruction is placed in the system message, before every task. A different task prompt follows each time.\n\n"
    "HOW THE INSTRUCTION IS USED\n"
    "- Each task prompt contains one rule that a program checks automatically, such as a limit on length, case, "
    "punctuation, keywords or format.\n"
    "- The program checks the model's whole reply. Anything the model writes in the reply, including reasoning, a plan "
    "or a self-check, is checked as part of the answer.\n"
    "- The system message above your instruction already defines the format of the reply: a header line before the "
    "answer and a footer line after it. Your instruction must not contradict that format, and must not ask for "
    "anything to appear before the header or after the footer.\n\n"
    "WHAT TO DO\n"
    "Write the best instruction you can. Think carefully before you write: work out the edge cases of each kind of "
    "rule (for example how words, sentences, paragraphs and keywords are counted, how case and punctuation rules "
    "apply to titles, quotes and names, and what exact output each rule needs), then turn what you learned into "
    "clear rules and worked examples inside the instruction. The small model must not write out its own reasoning.\n"
    "You may search the web for good prompt-writing guidance and for how this kind of rule-following benchmark "
    "(IFEval) is checked. List any pages you actually used.\n\n"
    "GUIDELINES FROM THE PROMPTING LITERATURE (follow them unless you have a good reason not to)\n"
    "1. Put the most important constraints and the output rules first.\n"
    "2. Use one consistent structure, such as short headings or tags, and stay consistent.\n"
    '3. Be direct and exact. Skip persuasive or emphatic words such as "absolutely" or "perfectly".\n'
    "4. Prefer positive phrasing: say what to do. When a rule is a prohibition, also state what to do instead.\n"
    "5. Include three to five short worked examples, each a task prompt and a correct reply, laid out identically. "
    "Vary the content of the examples, not their structure, and do not use many: too many examples make a model "
    "copy them. Build them from the samples below.\n"
    "6. Do not ask the small model for visible step-by-step reasoning or a visible self-check. Studies of "
    'instruction following (for example "When Thinking Fails", 2025) found that reasoning text often lowers '
    "accuracy on rule-following tasks, especially for small models. Turn any checks into rules about the final reply.\n"
    "7. The length is up to you. Long, wordy instructions written by strong models can overwhelm small models, so "
    "make every sentence carry a rule or an example.\n"
    '8. Assume strict checking of the raw reply: no preamble such as "Here is", and nothing after the answer.\n\n'
    "THE SAMPLES\n"
    "Below are 24 samples, one for each kind of rule. Each shows the rule the program checks, followed by a task "
    "prompt that contains it.\n\n"
    "{samples}\n\n"
    "OUTPUT\n"
    "Reply with the instruction text only. If you used web pages, add a line containing only ---SOURCES--- after "
    "the instruction, then one URL per line."
)

# where the writer guidelines above come from (a research summary written on 2026-11-10; points marked in reports/M5_v4_plan.md)
WRITER_GUIDELINE_SOURCES = [
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices",
    "https://aistudio.google.com/docs/prompting-strategies",
    "https://developers.openai.com/api/docs/guides/prompt-engineering",
    "https://huggingface.co/papers/2505.11423",
    "https://arxiv.org/html/2505.09930v4",
    "https://arxiv.org/html/2311.07911v1",
]
