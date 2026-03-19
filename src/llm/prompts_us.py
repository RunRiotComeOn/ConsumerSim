"""
US version prompt templates for LLM-based survey response generation.
English prompts with US-specific context (race, party, healthcare, student debt).
"""
from typing import List, Optional
from src.agents.base_us import USConsumerAgent


# SYSTEM_PROMPT_US = """\
# You are simulating American consumers responding to the University of Michigan \
# Survey of Consumers (the source of the Michigan Consumer Sentiment Index).

# Each consumer profile contains demographic data AND subjective perceptions \
# (happiness, financial satisfaction, future confidence, social trust, etc.).

# Interpret each consumer profile as that person's BASELINE condition: their \
# usual demographic situation, finances, and underlying attitudes in the absence \
# of a new short-term monthly shock.

# Separately, you may also receive CURRENT MONTH SHOCKS or real-time economic \
# conditions. These are temporary disturbances that can push the consumer away \
# from their baseline for this month only.

# When answering the survey:
# 1. Start from the consumer's baseline profile.
# 2. Adjust that baseline using the current month's shocks.
# 3. Then give the final answers for this month.

# IMPORTANT US-specific factors to consider:
# - Political leaning strongly affects economic outlook: partisans of the \
# current president's party tend to be much more optimistic about business \
# conditions (BUS12, BUS5), sometimes by 30-40 points.
# - Race matters: Black Americans historically report lower consumer sentiment \
# than White Americans (20-30 point ICS gap), partly reflecting structural \
# economic inequality.
# - Homeownership status affects sensitivity to interest rates and housing \
# market conditions.
# - Student debt burden weighs on young consumers' personal financial outlook.
# - Health insurance status affects economic security perceptions.
# - Gas prices are a uniquely strong predictor of US consumer sentiment.

# Use ALL the subjective indicators to infer how they would answer. For example:
# - A person with LOW financial satisfaction and no health insurance likely feels "worse off"
# - A Republican under a Democratic president with HIGH income may still say business \
# conditions are "bad" due to partisan perception
# - A young person with student debt but HIGH future confidence may say personal \
# finances will be "better"
# - A severe monthly shock can make even a normally stable consumer more pessimistic \
# for this month, especially on DUR, BUS12, and BUS5

# The effect of a shock should depend on the consumer's exposure and vulnerability.
# For example:
# - Inflation, rates, and affordability shocks matter more for DUR and household finances
# - Labor market shocks matter more for PAGO, PEXP, BUS12, and BUS5
# - Trade or tariff shocks can worsen national business outlook, especially for BUS12 and BUS5

# Do NOT rely solely on demographics. The subjective perception data is the most \
# informative signal for predicting survey responses.

# Output ONLY a valid JSON array. No explanations, no markdown code blocks, no extra text.
# """

SYSTEM_PROMPT_US = """\
You are role-playing as individual American consumers being surveyed by the \
University of Michigan Survey of Consumers.

For each consumer profile you receive, imagine you ARE that person — with their \
income, age, education, political leaning, race, region, debts, and subjective \
feelings (happiness, financial satisfaction, confidence, trust, etc.).

You will also receive CURRENT ECONOMIC CONDITIONS and NEWS HEADLINES that this \
consumer has been exposed to in daily life during the survey period.

---

## HOW TO THINK (step by step, internally)

For EACH consumer, put yourself in their shoes:

1. **Who am I?** Read the demographic profile. Understand my life situation — \
my income, job security, debts, whether I own a home, my political views.

2. **How was I feeling before this month?** Use the subjective perception data \
(financial satisfaction, happiness, future confidence) as my pre-existing mood.

3. **What have I seen in the news this month?** Read the headlines and economic \
data. Think about how each piece of news affects ME PERSONALLY given my \
specific situation. Not all news affects all people equally:
   - A homeowner cares more about mortgage rates than a renter.
   - A high-income investor reacts to stock market news more than a minimum-wage worker.
   - A young person with student debt worries about different things than a retiree.
   - Political leaning colors how I interpret the same economic news.

4. **How do I answer the survey?** Based on who I am + what I've seen, give \
honest answers. Answer like a real person would — gut feeling, not analysis.

---

## KEY BEHAVIORAL PATTERNS IN REAL SURVEY DATA

* **Personal finance questions (PAGO, PEXP)** are driven more by personal \
situation — income, job, debts, costs I face daily. News has a moderate effect.

* **National outlook questions (BUS12, BUS5)** are driven heavily by news \
AND political leaning. These questions show the most variation.

* **Buying conditions (DUR)** are sensitive to prices, interest rates, and \
whether consumers feel financially stretched.

* **Political leaning matters**: Partisans of the party NOT in power tend to \
be more pessimistic about business conditions. This is one of the strongest \
predictors in the real survey data.

* **Not everyone reacts the same way to news**: Some consumers are resilient \
(high income, stable job, optimistic personality). Others are vulnerable \
(low income, indebted, anxious). The SAME news headline should produce \
DIFFERENT answers from different consumers.

---

## IMPORTANT: DIVERSITY OF RESPONSES

A batch of consumers should show GENUINE DIVERSITY in answers — not everyone \
says the same thing. Even in a bad economy, some people are doing fine \
personally. Even in a good economy, some people are struggling.

Do NOT default to "same"/"uncertain" for everyone — real consumers have \
opinions. But also do not make everyone negative or everyone positive. \
Let each consumer's unique profile drive their individual answer.

---

## OUTPUT FORMAT

Return ONLY a valid JSON array. No explanations, no markdown, no extra text.

Each object must have exactly: PAGO, DUR, PEXP, BUS12, BUS5

Allowed values:
* PAGO, PEXP: "better" | "same" | "worse"
* DUR, BUS12, BUS5: "good" | "uncertain" | "bad"
"""

def build_batch_prompt_us(agents: List[USConsumerAgent],
                          macro_context: str = "",
                          prediction_target_month: Optional[str] = None,
                          survey_cutoff_date: Optional[str] = None,
                          prior_ics: Optional[float] = None) -> str:
    """Build a batch prompt for N US agents, optionally with real macro context."""
    timing_lines = []
    if prediction_target_month:
        timing_lines.append(f"- Target survey month: {prediction_target_month}")
    if survey_cutoff_date:
        timing_lines.append(f"- Information cutoff date: {survey_cutoff_date}")
    if not timing_lines:
        timing_lines.append("- Target survey month: current month")
        timing_lines.append("- Information cutoff date: use only information available by survey time")

    timing_block = (
        "--- PREDICTION TIMING ---\n"
        + "\n".join(timing_lines)
        + "\nTreat this as a real-time prediction at the cutoff date. Do NOT use events after the cutoff.\n"
        "--- END PREDICTION TIMING ---\n\n"
    )

    # Prior context: give model soft background about last month's public mood
    regime_block = ""
    if prior_ics is not None:
        if prior_ics < 60:
            mood = "deeply pessimistic — most people felt the economy was in serious trouble"
        elif prior_ics < 70:
            mood = "quite negative — widespread worry about the economy"
        elif prior_ics < 80:
            mood = "somewhat negative — more pessimism than optimism among the public"
        elif prior_ics < 90:
            mood = "mixed — people had varied views, leaning slightly cautious"
        else:
            mood = "fairly positive — most people felt okay about the economy"

        regime_block = (
            f"--- BACKGROUND: LAST MONTH'S PUBLIC MOOD ---\n"
            f"Last month, the overall mood among American consumers was {mood}.\n"
            f"Use this as context for where consumers are starting from this month. "
            f"Then decide whether THIS month's news and conditions make things better, "
            f"worse, or about the same compared to last month.\n"
            f"--- END BACKGROUND ---\n\n"
        )

    macro_section = ""
    if macro_context:
        macro_section = (
            f"{timing_block}"
            f"{regime_block}"
            f"--- CURRENT ECONOMIC CONDITIONS AND NEWS (what consumers have seen) ---\n"
            f"{macro_context}\n"
            f"--- END CURRENT CONDITIONS ---\n\n"
            f"Each consumer below has a baseline profile (who they are, how they normally feel). "
            f"Now imagine each consumer has been reading the news above and experiencing these "
            f"economic conditions. Answer the survey AS THAT PERSON would, given what they've seen.\n\n"
        )
    else:
        macro_section = (
            f"{timing_block}"
            f"{regime_block}"
            "No specific economic news is provided for this month. "
            "Answer based on the consumer's baseline profile only.\n\n"
        )

    survey_questions = """\
Survey Questions (University of Michigan Survey of Consumers):
1. PAGO: We are interested in how people are getting along financially these \
days. Would you say that you are better off or worse off financially than you \
were a year ago?
   → Answer: "better" | "same" | "worse"

2. DUR: About the big things people buy for their homes — such as furniture, \
a refrigerator, stove, television, and things like that. Generally speaking, \
do you think now is a good or bad time for people to buy major household items?
   → Answer: "good" | "uncertain" | "bad"

3. PEXP: Now looking ahead — do you think that a year from now you will be \
better off financially, or worse off, or just about the same as now?
   → Answer: "better" | "same" | "worse"

4. BUS12: Now turning to business conditions in the country as a whole — do \
you think that during the next twelve months we'll have good times financially, \
or bad times, or what?
   → Answer: "good" | "uncertain" | "bad"

5. BUS5: Looking ahead, which would you say is more likely — that in the \
country as a whole we'll have continuous good times during the next five years \
or so, or that we will have periods of widespread unemployment or depression, \
or what?
   → Answer: "good" | "uncertain" | "bad"
"""

    profiles_text = ""
    for i, agent in enumerate(agents):
        profiles_text += f"\n=== Consumer {i + 1} ===\n"
        profiles_text += agent.to_profile_text()
        profiles_text += "\n"

    output_format = f"""\
Return a JSON array with exactly {len(agents)} objects, one per consumer, in order.
Each object must have exactly these keys: PAGO, DUR, PEXP, BUS12, BUS5.

Example (for 2 consumers):
[
  {{"PAGO": "better", "DUR": "good", "PEXP": "better", "BUS12": "good", "BUS5": "good"}},
  {{"PAGO": "worse", "DUR": "bad", "PEXP": "same", "BUS12": "uncertain", "BUS5": "uncertain"}}
]
"""

    return f"{macro_section}{survey_questions}\n{profiles_text}\n{output_format}"
