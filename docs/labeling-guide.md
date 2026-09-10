# Labelling guide — the evaluation set

## The golden set: how it was sampled and labelled (the required note)

**File:** `eval/labelset/labels.amazon.jsonl` — **200 AmazonHelp messages**,
hand-labelled by me.

**Sampling.** `python -m eval.make_labelset --data data/amazonhelp.jsonl
--unlabelled --target 200` takes the first customer message of 200 AmazonHelp
threads, shuffled with a fixed seed, keeping only messages ≥ 3 words. No further
stratification — the set therefore mirrors AmazonHelp's real intent mix
(delivery-heavy: 86/200 `delivery_issue`, and ~10% non-English), which is what we
want to evaluate on. The cost is that `cancellation` (3), `account_access` (7)
and `order_status` (8) are thin; per-class numbers for those are noisy and the
report says so.

**Labelling.** Every row was read once against the rubric below and assigned
`intent` + `should_escalate` + a one-line `escalation_reason`. A second pass a
few hours later diffed against the first; disagreements (≈ 6%, mostly "angry vent
vs actionable complaint" and "product question vs needs-authority") were
re-decided and noted. 30% of rows are marked `split: "dev"` for threshold
tuning; the other 70% (`"test"`) produce every headline number. Rows are marked
`notes: "hand-labelled ... review before trusting"` — a real deployment would
have a second annotator and an inter-annotator-agreement figure.

**Reproduce the scores:** `python -m eval.report --labelset
eval/labelset/labels.amazon.jsonl --data data/amazonhelp.jsonl --model
models/clf.real.joblib`.

---

## Rubric

The synthetic set (`eval/labelset/labels.jsonl`, generator-truth labels) uses the
same schema, one JSON object per line:

```json
{
  "id": "syn-0421",
  "brand": "NimbusAir",
  "text": "Where is my order 48213391? It's been 6 days.",
  "intent": "order_status",
  "should_escalate": false,
  "escalation_reason": "routine, low/medium risk",
  "history": null,
  "ideal_reply_notes": "give ETA or tracking link; ask for order # only if not provided",
  "notes": "",
  "split": "test"
}
```

## How it is produced

- **Synthetic data (default):** `make labelset` reads the generator's recorded
  true intent and derives `should_escalate` from an explicit policy (see below).
  These rows are marked `notes: "auto-seeded ... review before trusting"`.
- **Real Kaggle data:** `python -m eval.make_labelset --data data/conversations.jsonl --unlabelled`
  samples ~200 threads, stratified by brand, and leaves `intent` / `should_escalate`
  blank for you to fill by hand.

## Labelling tool

`python -m eval.label_cli --in eval/labelset/labels.real.unlabelled.jsonl --out eval/labelset/labels.real.jsonl --limit 60`

shows one message at a time; press `1`–`0` for the intent, `y`/`n` for
`should_escalate`, `s` skip, `b` back, `q` save+quit. Resumable — re-run and it
skips rows already labelled. `--limit` lets you work in batches, then
`python -m eval.report --labelset eval/labelset/labels.real.jsonl --data data/conversations.jsonl --model models/clf.real.joblib`
scores against what you have so far.

The committed `labels.real.jsonl` (first 60 rows) was labelled by reading each
message against this guide; treat it as a starting point and correct anything
you disagree with — the borderline calls are noted per row.

Target size: **200** (assignment asks 150–250). Stratified so every intent has
≥ 15 examples. 30% of rows are tagged `split: "dev"` for threshold tuning; the
other 70% (`"test"`) produce every headline number and must not be looked at
while tuning.

## Intent — pick exactly one

Use the definitions in `src/support_agent/taxonomy/intents.py`. Tie-break rules:

1. **Action the customer wants > topic they mention.** "I was charged for a
   plan I cancelled" is `billing_dispute` (they want the charge fixed), not
   `cancellation` (already done).
2. **Problem > sentiment.** An angry message about a late parcel is
   `delivery_issue`, not `complaint_feedback`. Reserve `complaint_feedback` for
   dissatisfaction not tied to one fixable transaction ("your app keeps getting
   worse").
3. **Concrete fault > question.** "Does export work? Mine does nothing" is
   `technical_bug`, not `product_question`.
4. **When genuinely torn between two,** label the more operationally urgent one
   and write both + your reasoning in `notes`. Those notes are cited in the
   report's "what the metrics miss" section.
5. `other_unclear` only when you could not route it to a human team either.

## should_escalate — the gold routing label

Mark `true` (a human must handle it) if **any** hold:

- Not written in English.
- Contains a legal / regulatory / dispute cue: chargeback, GDPR/CCPA, lawyer,
  regulator, ombudsman, "delete my account", "dispute the charge".
- Intent is high-risk: `billing_dispute`, `cancellation`, `account_access`.
- The customer is clearly angry **and** the issue is non-trivial (not
  `praise_thanks` / simple `product_question`).
- Answering correctly needs information or authority the agent cannot have
  (identity verification, a policy exception, a refund approval).

Otherwise `false`. Record the deciding factor in `escalation_reason` in your own
words — that text is not scored, but it is read during failure analysis.

## ideal_reply_notes — optional but useful

One line on what a good reply must do (not verbatim wording). Used when reading
generation failures, e.g. "must not promise a refund amount; should ask for the
order number". Leave blank if obvious.

## Quality bar

- Label in one pass without looking at model output; do a second pass a day
  later and diff — disagreements with yourself are the real inter-annotator
  signal here and belong in `notes`.
- If a row is unusable (truncated, not actually a support message), delete the
  line rather than forcing a label.
