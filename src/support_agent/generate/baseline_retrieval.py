"""Baseline #2: return the nearest resolved reply verbatim, no LLM.

If RAG generation cannot beat copy-pasting the single closest historical reply
(on judge score and groundedness), the LLM is not earning its place in the
pipeline. This function is what the generator is measured against.
"""

from __future__ import annotations

from ..types import Exemplar, GeneratedReply


def nearest_reply_baseline(exemplars: list[Exemplar]) -> GeneratedReply:
    if not exemplars:
        return GeneratedReply(
            text="",
            exemplars=[],
            used_llm=False,
            model="retrieval_baseline",
            grounded=True,
            groundedness_notes=["no exemplars"],
        )
    top = exemplars[0]
    return GeneratedReply(
        text=top.agent_text,
        exemplars=exemplars,
        used_llm=False,
        model="retrieval_baseline",
        grounded=True,  # it is a real historical reply, but may be for a different case
        groundedness_notes=[f"verbatim reply from {top.conv_id} (sim={top.similarity:.2f})"],
    )
