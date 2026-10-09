"""Personas and prompt templates for every LLM system in the pipeline."""

from __future__ import annotations

PERSONAS: list[dict[str, str]] = [
    {
        "name": "persona_far_left",
        "short": "a 24-year-old far-left union organizer",
        "system": (
            "You are a 24-year-old union organizer living in Minneapolis, "
            "politically on the far left of the Democratic party, closer to "
            "Bernie Sanders than to the party establishment. You get your "
            "news from Jacobin, The Intercept, and political social media "
            "accounts. Answer as this person genuinely would, not as an AI "
            "assistant."
        ),
    },
    {
        "name": "persona_center_left",
        "short": "a 45-year-old mainstream Democratic high school teacher",
        "system": (
            "You are a 45-year-old high school teacher in a suburb of "
            "Philadelphia, a mainstream Democratic voter with no strong ties "
            "to the party's left or center-right factions. You get your news "
            "mostly from NPR and the New York Times. Answer as this person "
            "genuinely would, not as an AI assistant."
        ),
    },
    {
        "name": "persona_centrist",
        "short": "a 41-year-old suburban parent with no party affiliation",
        "system": (
            "You are a 41-year-old suburban parent with no strong party "
            "affiliation, who skims news headlines once a day and rarely "
            "reads past the first paragraph. Answer as this person genuinely "
            "would, not as an AI assistant."
        ),
    },
    {
        "name": "persona_center_right",
        "short": "a 52-year-old mainstream Republican dealership manager",
        "system": (
            "You are a 52-year-old car dealership manager in an exurban town "
            "in Ohio, a mainstream Republican voter, not aligned with the "
            "party's populist wing. You get your news from the Wall Street "
            "Journal and local TV news. Answer as this person genuinely "
            "would, not as an AI assistant."
        ),
    },
    {
        "name": "persona_far_right",
        "short": "a 58-year-old rural small-business owner and firm Trump supporter",
        "system": (
            "You are a 58-year-old small-business owner in a rural US "
            "county, a firm Trump supporter, who mostly gets news from "
            "conservative talk radio and local TV. Answer as this person "
            "genuinely would, not as an AI assistant."
        ),
    },
    {
        "name": "persona_disengaged",
        "short": "a 29-year-old delivery driver who does not follow politics",
        "system": (
            "You are a 29-year-old delivery driver in Atlanta working long, "
            "irregular hours, who does not follow politics or the news at "
            "all. Whatever you know about current events comes from "
            "overhearing friends or scrolling social media by chance. "
            "Answer as this person genuinely would, not as an AI assistant."
        ),
    },
]

# One reader with no ideological or demographic prior, for the self-consistency ablation.
GENERIC_READER: dict[str, str] = {
    "name": "reader_generic",
    "short": "an ordinary reader",
    "system": (
        "You are an ordinary adult who follows the news from time to time. "
        "Answer as this person genuinely would, not as an AI assistant."
    ),
}

STANCE_VARIANTS: list[str] = [
    """Read the article below the way the person you are would, then answer with ONLY a JSON object and nothing else, in exactly this form:
{"reaction": "<one short sentence, in character>", "stance": <int>}

stance meaning: -1 disagrees with or is angered by the article's principal claim, 0 is neutral or unmoved, 1 agrees with or is reassured by it.

Title: <<TITLE>>
Article (lead only): <<LEAD>>""",
    """Here is a news article. React to it as you, in your own situation, naturally would. Reply with only a JSON object and no other text, shaped like this:
{"reaction": "<one short sentence in your own voice>", "stance": <int>}

Use stance -1 if you disagree with the article's main claim or it angers you, 0 if you feel neutral or unmoved, and 1 if you agree with it or it reassures you.

Title: <<TITLE>>
Lead: <<LEAD>>""",
    """You just came across the following news story. Tell me how it lands with you. Output a single JSON object and nothing else, formatted exactly as {"reaction": "<one short sentence, in character>", "stance": <int>}, where stance is -1 (you reject the article's central claim or it upsets you), 0 (you are neutral or unmoved) or 1 (you accept the central claim or it reassures you).

Headline: <<TITLE>>
Opening paragraph: <<LEAD>>""",
    """Read this article as the person you are, then respond with JSON only. Required format:
{"reaction": "<one short sentence, in character>", "stance": <int>}
Stance scale: -1 means you disagree with the main claim or feel angry about it. 0 means you have no strong feeling either way. 1 means you agree with the main claim or feel reassured by it.

Article title: <<TITLE>>
Article text (first 150 words): <<LEAD>>""",
    """Imagine this article shows up in your news feed. Give your honest reaction, in character. Your whole answer must be one JSON object in exactly this shape:
{"reaction": "<one short sentence, in character>", "stance": <int>}
The stance must be one of: -1 (you disagree with, or are angered by, the article's principal claim), 0 (neutral or unmoved), 1 (you agree with, or are reassured by, the principal claim).

Title: <<TITLE>>
Lead paragraph: <<LEAD>>""",
]

# The annotation guideline, word for word, shared by the human annotators, the silver annotator and
# the first paraphrase of every prompt that judges controversy.
GUIDELINE_Q1 = """Question: Does the passage discuss a topic or event that generates widespread moral, political, or ideological debate among the public?

YES
Definition: The subject matter involves actions, policies, decisions, or events that actively divide public or political opinion into sharply opposing camps.
Indicators:
- Military interventions, state-sanctioned strikes, or targeted assassinations.
- Highly polarized social, political, economic, or environmental legislation.
- Debates with no broad societal consensus where both sides hold substantial influence.
Note: Even if the text presents the event through neutral or journalistic reporting, the underlying topic still qualifies as YES if society is polarized on it.

NO
Definition: The subject matter covers non-divisive facts, neutral logistics, shared humanitarian concerns, or topics governed by a general societal consensus.
Indicators:
- Broadly accepted historical facts or timeline recaps without disputed agency.
- Routine administrative announcements, scientific facts, or general consensus issues.
- Recognition of objective danger (e.g., agreeing that war is destructive does not constitute a divided controversy)."""

GUIDELINE_Q2 = """Question: How clearly does the text provide enough context to make this judgment?
Select the option that reflects how straightforward it was to apply the criteria above to the text.

HIGH
Definition: The topic and its context are unmistakable.
Criteria: The text explicitly names entities, actions, or events with well-established societal or geopolitical divides (or obvious consensus), leaving no ambiguity about public polarization.

MEDIUM
Definition: Plausible topic identification, but some ambiguity remains.
Criteria: The subject is recognizable, but the passage focuses purely on personal diplomacy, logistics, neutral chronology, or contains clipped text that partially obscures the wider debate.

LOW
Definition: High uncertainty or insufficient evidence.
Criteria: The excerpt is too short, heavily redacted, or mentions the issue so tangentially that deciding whether the topic is genuinely controversial requires guessing or external speculation."""

GUIDELINE_EXAMPLES = """Edge cases and examples:
- Factual reporting on polarizing events. Example: reporting on the strike against Qassem Suleimani and subsequent regional fallout. Controversy: YES (the act itself and US-Iran relations are globally divisive). Confidence: HIGH (the event and political figures are clearly named).
- Neutral statements about risk or danger. Example: a leader stating they hope to avoid civilian infrastructure damage during a crisis. Controversy: NO (desiring safety or acknowledging risk is a universal concern, not a polarized debate). Confidence: MEDIUM or HIGH depending on whether the underlying conflict overshadows the statement."""

SCORE_EXPLAINED = (
    "Instead of a YES or NO label, give a score between 0 and 1 for how clearly the answer to the question is YES. "
    "1 means the topic clearly divides the public into sharply opposing camps, 0 means it is clearly non-divisive "
    "or governed by a general consensus, and values in between express doubt."
)

BASELINE_VARIANTS: list[str] = [
    f"""You are annotating a news article. Follow these annotation guidelines exactly.

{GUIDELINE_Q1}

{GUIDELINE_EXAMPLES}

{SCORE_EXPLAINED}

Answer with ONLY a JSON object and nothing else, in exactly this form:
{{"controversy": <number between 0 and 1>, "reason": "<one short sentence>"}}

Title: <<TITLE>>
Article (lead only): <<LEAD>>""",
    """Rate the news article below with this guideline in mind.

Controversy question: does the passage discuss a topic or event that generates widespread moral, political, or ideological debate among the public?
It is YES when actions, policies, decisions, or events actively divide public or political opinion into sharply opposing camps. Typical cases are military interventions, state-sanctioned strikes or targeted assassinations, highly polarized social, political, economic, or environmental legislation, and debates with no broad societal consensus where both sides hold substantial influence. A neutral, journalistic tone does not change this: a polarizing topic reported calmly is still YES.
It is NO when the subject is non-divisive facts, neutral logistics, shared humanitarian concerns, or a topic governed by general societal consensus. Typical cases are broadly accepted historical facts or timelines without disputed agency, routine administrative announcements, scientific facts, and recognition of objective danger (agreeing that war is destructive is not a divided controversy).

Reference cases: reporting on the strike against Qassem Suleimani and its regional fallout is YES. A leader saying they hope to avoid civilian infrastructure damage during a crisis is NO, because desiring safety is a universal concern and not a polarized debate.

Give a score between 0 and 1 for how clearly the answer is YES: 1 means the topic clearly splits the public into opposing camps, 0 means it is clearly non-divisive or a matter of consensus, and values in between express doubt.

Reply with ONLY a JSON object, no other text:
{"controversy": <number between 0 and 1>, "reason": "<one short sentence>"}

Title: <<TITLE>>
Lead: <<LEAD>>""",
    """Your task is to judge whether the news excerpt below (title and lead) concerns a controversial topic.

A topic is controversial (score near 1) if it generates widespread moral, political, or ideological debate and splits public or political opinion into sharply opposing camps. Indicators: military interventions, strikes, targeted assassinations; highly polarized legislation (social, political, economic, environmental); debates with no broad consensus where both sides are influential. The wording of the article does not matter. If society is polarized on the underlying topic, the topic is controversial even when the article is neutral.
A topic is not controversial (score near 0) if it covers non-divisive facts, neutral logistics, shared humanitarian concerns, or matters of general consensus: accepted historical facts, routine administrative news, scientific facts, or the recognition of objective danger (saying war is destructive does not make a divided controversy).

Calibration: reporting on the Qassem Suleimani strike and its fallout is controversial. A leader hoping to avoid damage to civilian infrastructure in a crisis is not, since that concern is universal.

Use values in between when you doubt. Output a single JSON object and nothing else, formatted exactly as {"controversy": <number between 0 and 1>, "reason": "<one short sentence>"}.

Headline: <<TITLE>>
Opening paragraph: <<LEAD>>""",
]

AGGREGATE_VARIANTS: list[str] = [
    f"""You are the judge in a panel of simulated news readers. Follow these annotation guidelines exactly to decide whether the article concerns a controversial topic.

{GUIDELINE_Q1}

{GUIDELINE_EXAMPLES}

Several simulated readers reacted to the article below. Stance -1 means the reader disagrees with or is angered by the article's main claim, 0 means neutral, 1 means agrees with or is reassured by it. Use their reactions as evidence about how divided the public is on the topic.

{SCORE_EXPLAINED}

Answer with ONLY a JSON object and nothing else, in exactly this form:
{{"summary": "<one or two sentences on where these readers actually disagree, or say plainly that they do not>", "controversy": <number between 0 and 1>}}

Title: <<TITLE>>

Reader reactions:
<<READERS>>""",
    """Below is a news headline and the reactions of several readers with different backgrounds. Each reaction comes with a stance: -1 (disagrees with or is angered by the article's main claim), 0 (neutral), 1 (agrees with or is reassured by it).

Decide with this guideline whether the topic is controversial. It is YES when the topic or event generates widespread moral, political, or ideological debate and actively divides public or political opinion into sharply opposing camps (military interventions, state-sanctioned strikes, targeted assassinations, highly polarized legislation, debates with no broad consensus where both sides hold substantial influence), even if the reporting is neutral. It is NO when the topic is non-divisive facts, neutral logistics, shared humanitarian concerns, or something society broadly agrees on, including the recognition of objective danger. Reporting on the strike against Qassem Suleimani is YES. A leader hoping to avoid damage to civilian infrastructure is NO.

Give a score between 0 and 1 for how clearly the answer is YES: 1 means a clearly divided public, 0 means clear consensus, in between means doubt. Reply with ONLY a JSON object, no other text:
{"summary": "<one or two sentences on where these readers actually disagree, or say plainly that they do not>", "controversy": <number between 0 and 1>}

Headline: <<TITLE>>

Reactions:
<<READERS>>""",
    """You judge a panel of simulated news readers. Their reactions to the article titled below are given with a stance: -1 disagrees or is angered, 0 neutral, 1 agrees or is reassured.

Your question is whether the article's topic generates widespread moral, political, or ideological debate among the public. Controversial topics split opinion into sharply opposing camps: military interventions, strikes, targeted assassinations, highly polarized legislation, open debates where both sides are influential. Neutral wording does not make such a topic less controversial. Non-controversial topics are non-divisive facts, neutral logistics, shared humanitarian concerns, and matters of general consensus, such as accepted history, routine announcements, scientific facts, or acknowledging objective danger. For calibration, the Qassem Suleimani strike is controversial, and a leader hoping to avoid civilian infrastructure damage is not.

Output a single JSON object and nothing else, formatted exactly as {"summary": "<one or two sentences on where these readers actually disagree, or say plainly that they do not>", "controversy": <number between 0 and 1>}. Use 1 when the answer is clearly YES, 0 when it is clearly NO, and values in between for doubt.

Article title: <<TITLE>>

The readers said:
<<READERS>>""",
]

DELIBERATION_VARIANTS: list[str] = [
    """You read the article below and reacted to it. Now you see how the other readers reacted. You may keep your stance or change it, as the person you are would.

Title: <<TITLE>>
Article (lead only): <<LEAD>>

Your first reaction (stance <<OWN_STANCE>>): <<OWN_REACTION>>

The other readers:
<<OTHERS>>

Answer with ONLY a JSON object and nothing else, in exactly this form:
{"reaction": "<one short sentence, in character>", "stance": <int>}
stance meaning: -1 disagrees with or is angered by the article's principal claim, 0 is neutral or unmoved, 1 agrees with or is reassured by it.""",
    """Earlier you reacted to the article below. Here is what some other readers said about it. Take it into account if it matters to you, then give your final reaction as the person you are. Keeping your first stance is fine.

Title: <<TITLE>>
Lead: <<LEAD>>

What you said first (stance <<OWN_STANCE>>): <<OWN_REACTION>>

What the others said:
<<OTHERS>>

Reply with only a JSON object and no other text, shaped like this:
{"reaction": "<one short sentence in your own voice>", "stance": <int>}
Use stance -1 if you disagree with the article's main claim or it angers you, 0 if you feel neutral or unmoved, and 1 if you agree with it or it reassures you.""",
    """Second round. You already reacted to this story; now you can see the reactions of other readers. Decide whether you want to revise your stance. You may also keep it unchanged. Output a single JSON object and nothing else, formatted exactly as {"reaction": "<one short sentence, in character>", "stance": <int>}, where stance is -1 (you reject the article's central claim or it upsets you), 0 (neutral or unmoved) or 1 (you accept the central claim or it reassures you).

Headline: <<TITLE>>
Opening paragraph: <<LEAD>>

Your first reaction (stance <<OWN_STANCE>>): <<OWN_REACTION>>

Other readers:
<<OTHERS>>""",
    """Read this article again as the person you are, now knowing how other readers reacted to it. Then respond with JSON only. Required format:
{"reaction": "<one short sentence, in character>", "stance": <int>}
Stance scale: -1 means you disagree with the main claim or feel angry about it. 0 means you have no strong feeling either way. 1 means you agree with the main claim or feel reassured by it. You are free to keep your first stance.

Article title: <<TITLE>>
Article text (first 150 words): <<LEAD>>

Your first reaction (stance <<OWN_STANCE>>): <<OWN_REACTION>>

Reactions of the other readers:
<<OTHERS>>""",
    """Imagine you posted your reaction to this article, then scrolled down and saw what other people wrote. Do you stay where you were or move? Answer honestly, in character. Your whole answer must be one JSON object in exactly this shape:
{"reaction": "<one short sentence, in character>", "stance": <int>}
The stance must be one of: -1 (you disagree with, or are angered by, the article's principal claim), 0 (neutral or unmoved), 1 (you agree with, or are reassured by, the principal claim).

Title: <<TITLE>>
Lead paragraph: <<LEAD>>

Your first reaction (stance <<OWN_STANCE>>): <<OWN_REACTION>>

What other people wrote:
<<OTHERS>>""",
]


SILVER_FORMAT = (
    'Answer with ONLY a JSON object and nothing else, in exactly this form:\n'
    '{"label": "YES" or "NO", "confidence": "HIGH" or "MEDIUM" or "LOW", "reason": "<one short line>"}\n'
    'Here "label" is your answer to the controversy question (is_controversial).'
)

SILVER_VARIANTS: list[str] = [
    f"""# Annotation guidelines: topic controversy and confidence assessment

Your task consists of two questions for each provided text:
1. Controversy (is_controversial): determine whether the core subject matter constitutes a polarized public issue.
2. Confidence (confidence): gauge the certainty of your decision based on the clarity and context provided in the excerpt.

## Metric 1: is the topic controversial?

{GUIDELINE_Q1}

## Metric 2: confidence level

{GUIDELINE_Q2}

## {GUIDELINE_EXAMPLES}

The text to annotate is the title and the 150-word lead below.

<<FORMAT>>

Title: <<TITLE>>
Lead: <<LEAD>>""",
]


def render(template: str, **values: str) -> str:
    """Fill the <<KEY>> placeholders of a template."""
    out = template
    for key, value in values.items():
        out = out.replace(f"<<{key}>>", value)
    return out


def fmt_stance(stance: int) -> str:
    """Write a stance as -1, 0 or +1."""
    return f"{stance:+d}" if stance else "0"


def _pick(variants: list[str], variant: int) -> str:
    if variant >= len(variants):
        raise ValueError(f"paraphrase {variant} requested but only {len(variants)} wordings exist for this prompt")
    return variants[variant]


def stance_prompt(variant: int, title: str, lead: str) -> str:
    """Instruction asking a persona for a reaction and a stance."""
    return render(_pick(STANCE_VARIANTS, variant), TITLE=title, LEAD=lead)


def baseline_prompt(variant: int, title: str, lead: str) -> str:
    """Instruction for the single agent that rates controversy directly."""
    return render(_pick(BASELINE_VARIANTS, variant), TITLE=title, LEAD=lead)


def reader_lines(readers: list[dict]) -> str:
    """List readers as '- description (stance s): reaction' lines."""
    return "\n".join(
        f"- {r['short']} (stance {fmt_stance(r['stance'])}): {r['reaction']}" for r in readers
    )


def aggregate_prompt(variant: int, title: str, readers: list[dict]) -> str:
    """Instruction for the aggregator that judges controversy from the reactions."""
    return render(_pick(AGGREGATE_VARIANTS, variant), TITLE=title, READERS=reader_lines(readers))


def deliberation_prompt(variant: int, title: str, lead: str, own: dict, others: list[dict]) -> str:
    """Second-round instruction: the persona sees the others' reactions and may revise."""
    return render(
        _pick(DELIBERATION_VARIANTS, variant),
        TITLE=title,
        LEAD=lead,
        OWN_STANCE=fmt_stance(own["stance"]),
        OWN_REACTION=own["reaction"],
        OTHERS=reader_lines(others),
    )


def silver_prompt(variant: int, title: str, lead: str) -> str:
    """Annotation instruction for the silver labeler (same guideline as the human annotators)."""
    return render(_pick(SILVER_VARIANTS, variant), TITLE=title, LEAD=lead, FORMAT=SILVER_FORMAT)
