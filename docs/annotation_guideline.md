# Annotation guidelines: topic controversy and confidence assessment

Annotators saw only the title and the first 150 words of each article, in random order, without the source, the stratum or the other labels. For each article they filled three columns: `label` (YES or NO), `confidence` (LOW, MEDIUM or HIGH) and `reason` (one short line).

The text below is the guideline used for the gold labels. It is also the exact text given to the silver annotator, and it is defined in `pipeline/prompts.py`.

Your task consists of two questions for each provided text:
1. Controversy (is_controversial): determine whether the core subject matter constitutes a polarized public issue.
2. Confidence (confidence): gauge the certainty of your decision based on the clarity and context provided in the excerpt.

## Metric 1: is the topic controversial?

Question: Does the passage discuss a topic or event that generates widespread moral, political, or ideological debate among the public?

### YES

Definition: The subject matter involves actions, policies, decisions, or events that actively divide public or political opinion into sharply opposing camps.

Indicators:

- Military interventions, state-sanctioned strikes, or targeted assassinations.
- Highly polarized social, political, economic, or environmental legislation.
- Debates with no broad societal consensus where both sides hold substantial influence.

Note: Even if the text presents the event through neutral or journalistic reporting, the underlying topic still qualifies as YES if society is polarized on it.

### NO

Definition: The subject matter covers non-divisive facts, neutral logistics, shared humanitarian concerns, or topics governed by a general societal consensus.

Indicators:

- Broadly accepted historical facts or timeline recaps without disputed agency.
- Routine administrative announcements, scientific facts, or general consensus issues.
- Recognition of objective danger (e.g., agreeing that war is destructive does not constitute a divided controversy).

## Metric 2: confidence level

Question: How clearly does the text provide enough context to make this judgment?

Select the option that reflects how straightforward it was to apply the criteria above to the text.

### HIGH

Definition: The topic and its context are unmistakable.

Criteria: The text explicitly names entities, actions, or events with well-established societal or geopolitical divides (or obvious consensus), leaving no ambiguity about public polarization.

### MEDIUM

Definition: Plausible topic identification, but some ambiguity remains.

Criteria: The subject is recognizable, but the passage focuses purely on personal diplomacy, logistics, neutral chronology, or contains clipped text that partially obscures the wider debate.

### LOW

Definition: High uncertainty or insufficient evidence.

Criteria: The excerpt is too short, heavily redacted, or mentions the issue so tangentially that deciding whether the topic is genuinely controversial requires guessing or external speculation.

## Edge cases and examples

- Factual reporting on polarizing events. Example: reporting on the strike against Qassem Suleimani and subsequent regional fallout. Controversy: YES (the act itself and US-Iran relations are globally divisive). Confidence: HIGH (the event and political figures are clearly named).
- Neutral statements about risk or danger. Example: a leader stating they hope to avoid civilian infrastructure damage during a crisis. Controversy: NO (desiring safety or acknowledging risk is a universal concern, not a polarized debate). Confidence: MEDIUM or HIGH depending on whether the underlying conflict overshadows the statement.
