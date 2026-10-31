"""Rules for when the bot should stop answering and hand off to a human.

Kept as plain keyword rules on purpose: easy to explain, easy to test, easy to extend.
They run before the model as a guardrail. The model can also escalate, by calling its
create_support_ticket tool, e.g. when the customer accepts the offer of a ticket ("yes please").
"""
import re

# (category, reason shown to the customer and stored on the ticket, patterns). First match wins.
# Category None = let the LLM ticket summary pick it (a "talk to a human" can be about anything).
TRIGGERS = [
    (None, "You asked to speak with a person", [
        r"\bhuman\b", r"\breal person\b", r"\brepresentative\b", r"\blive agent\b",
        r"\btalk to (someone|somebody|an? (agent|person))\b", r"\bspeak to (someone|somebody|an? (agent|person))\b",
        r"\b(didn'?t|did not|doesn'?t|does not) help\b", r"\bnot helpful\b", r"\bunhelpful\b",
    ]),
    ("billing", "I'm not able to resolve billing issues directly", [
        r"\bcharged (twice|two times|double)\b", r"\bdouble[- ]charged\b", r"\bovercharged\b",
        r"\bbilling\b", r"\bpayment (failed|declined)\b", r"\bwrong (amount|charge)\b",
    ]),
    ("account", "I'm not able to resolve account access issues directly", [
        r"\blocked out\b", r"\b(can'?t|cannot|unable to) (log ?in|sign ?in|access my account)\b",
        r"\bhacked\b", r"\bcompromised\b",
    ]),
    ("complaint", "I'm sorry about your experience and want the right person to help", [
        r"\bcomplaint\b", r"\bridiculous\b", r"\bworst\b", r"\bfurious\b", r"\bunacceptable\b",
        r"\bterrible service\b", r"\bangry\b",
    ]),
]
# Note: "refund" / "return" are deliberately NOT triggers, so policy questions like
# "what is your refund policy?" are still answered from the knowledge base.


def check_triggers(text: str) -> tuple[str | None, str] | None:
    """Return (category, reason) if the message should be escalated, else None."""
    for category, reason, patterns in TRIGGERS:
        if any(re.search(p, text, re.I) for p in patterns):
            return category, reason
    return None
