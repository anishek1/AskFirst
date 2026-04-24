Ask First - AI Intern Assignment Duration: 3 days

Context
Ask First is a health clarity platform. Our AI companion Clary remembers user health history
across conversations, identifies patterns over time, and connects dots users themselves don't
notice. Your task is to build the reasoning layer that makes this possible.

The dataset
You will receive a synthetic JSON dataset of 3 user profiles. Each user has 8-10 health
conversations across 3 months. Conversations are timestamped and messy, users describe
symptoms inconsistently, mention lifestyle details casually, and don't connect events
themselves.
There are 8 hidden patterns planted deliberately across the 3 profiles. Your system needs to find
as many as possible without being told what to look for.

What to build
A system: Streamlit, Gradio, CLI, anything runnable that does two things:

1) One Cross conversation pattern detection with temporal reasoning
Read the full user history and surface hidden health patterns with causal and time-aware
reasoning.
Example of what we want:
"User 2 reported hair fall in Session 9 (Week 11). In Session 3 (Week 2), user mentioned
starting intermittent fasting under 800 calories daily. Hair fall typically manifests 8-12 weeks
after nutritional deficiency onset. These events are likely connected."
Example of what we don't want:
"User mentioned hair fall. User mentioned a diet change."

The system must reason across time — not just retrieve keywords. A symptom appearing 10
weeks after a lifestyle change means something completely different from a symptom appearing
10 weeks before one.

2) Two — Confidence scoring per pattern
For each pattern found, output a confidence score with a one line justification in json format ,
output strictly needs to be json format and all responses must support the stream & Why does
the system think this connection is real and not coincidental?
Example:
"Stomach pain → late night eating.
Confidence: high
Pain reported in Sessions 1, 4, 7 — all within 48 hours of late eating mentions in preceding
sessions. No pain reported in sessions without late eating mentions."
This forces explicit reasoning not just association.

Tech requirements
Any LLM. Document your choice and why.
No hardcoded patterns. Everything must be generated dynamically from the data.
Show your reasoning trace, not just the final output. We want to see what the system considers
before reaching a conclusion.
Document your chunking and context management strategy. How you decide what past context
to include in each reasoning call matters as much as the output itself.

What to submit
GitHub repo with working code and README covering setup and how to run it.

One page writeup mandatory, not optional-
1. Covering three things: your approach to the reasoning problem

2. Where your system fails or hallucinates confidently, and what you would build differently
with more time.
A submission with honest failure analysis scores higher than one that only shows the happy

path.

Scoring
Criteria                          Weight
Hidden patterns found out of 8      40%
Temporal reasoning quality           30%
Loopholes and improvement
thinking                             30%