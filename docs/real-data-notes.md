# Real-data run — notes

The committed metrics in [`report.md`](report.md) are on the synthetic sample
corpus, because that is the only path with gold labels that runs in CI with no
credentials. This file records what happened when the pipeline was pointed at the
**real Kaggle dataset** (`thoughtvector/customer-support-on-twitter`).

Reproduce:

```bash
KAGGLE_API_TOKEN=KGAT_...  make data      # -> data/conversations.jsonl (40k threads)
python -m support_agent.taxonomy.discover --data data/conversations.jsonl
python -m support_agent.train --data data/conversations.jsonl --out models/clf.real.joblib
python -m eval.make_labelset --data data/conversations.jsonl --unlabelled   # seed to hand-label
```

## 1. Dataset shape (40,000 threads, capped from ~3M tweets)

| | |
|---|---|
| Conversations threaded | 40,000 |
| Marked `resolved` (agent had last turn) | 33,174 (83%) |
| Distinct brands | 107 |
| Mean turns / thread | 3.3 |
| Biggest brands | AppleSupport (4.3k), AmazonHelp (4.0k), Uber_Support (2.0k), SpotifyCares, UPSHelp, Delta, AmericanAir, comcastcares, TMobileHelp |

The brand mix is dominated by **tech, telecom, airlines, and retail/logistics** —
domains the synthetic corpus does not model (no airline delay scenarios, no
telecom porting, no streaming-app bugs).

## 2. Taxonomy discovery on real data

`taxonomy/discover.py` on a 4,000-message sample chose k=16 with a **silhouette of
0.036** (vs ~0.40 on synthetic). Real support text does not cluster cleanly —
expected. The clusters still map recognisably onto the frozen 10 intents:

| Cluster theme (top terms) | Maps to |
|---|---|
| account · card · refund · money | `billing_dispute` |
| delivery · package · order · delivered · day | `delivery_issue` / `order_status` |
| account · phone · email · login · locked out | `account_access` |
| iphone · ios · update · battery · slow | `technical_bug` (devices) |
| game · xbox · hulu · spotify · app · playback | `technical_bug` (apps) |
| flight · delayed · plane · gate | *(travel-specific; currently splits across `delivery_issue` / `complaint_feedback`)* |
| customer service · worst · time | `complaint_feedback` |
| thank · love · christmas · favourite | `praise_thanks` |
| "help" · profanity · "dm" | `other_unclear` |

**Takeaway:** the taxonomy holds, but a real deployment on this data would add a
travel/trip-disruption intent and probably split `technical_bug` into
device vs service.

## 3. Weak labelling collapses on real text

| | Synthetic | Real |
|---|---|---|
| Rows the weak labeller could tag | ~72% | **~16%** (5,900 / 37,405; 31,505 abstained) |
| Classifier CV macro-F1 (on weak labels) | 0.99 | **0.70** |
| Worst-covered classes | — | `product_question` (25 rows), `delivery_issue` (57), `cancellation` (84) |
| Dominant class | balanced | `praise_thanks` (2,278) — 4× the next |

This is the headline real-data finding. Hand-rolled keyword rules are tuned to the
synthetic templates; real tweets say "where the hell is my stuff" and
"ported out 2 months ago still getting billed" and the rules abstain. Consequences
visible downstream:
- the real-data classifier over-predicts `praise_thanks` (its biggest training
  class) on ambiguous input;
- `delivery_issue` / `product_question` are essentially untrained.

**Fix path (unchanged from the report's roadmap, now with evidence):** more
labelling functions, a hand-labelled seed set per thin class, or a fine-tuned
encoder that needs far fewer examples per class. This is the first thing to do
before trusting real-data numbers.

## 4. Real end-to-end walkthrough (real data + real Gemini `flash-lite`)

Full `SupportAgent.triage` (real-data classifier, retrieval over 40k real
threads, `gemini-flash-lite-latest` for the draft). Unedited output:

| Brand / message | Intent (conf) | Retrieval sim | Decision | Draft reply |
|---|---|---|---|---|
| **AmazonHelp** — "parcel meant to arrive on the 28th, still not here, need it this week" | order_status (0.45) | 0.80 | **escalate** — low confidence | "I'm sorry your parcel hasn't arrived yet. What does your current tracking information say? You can check that here: `<link>` ^EA" |
| **SpotifyCares** — "upgraded to Premium but it's still shuffling, how do I turn that off?" | praise_thanks (0.29) ❌ | 0.76 | **escalate** — low confidence | "Hi there! Help's here. Can you DM us your account's username or email address? We'll check things backstage /AL" |
| **Delta** — "flight delayed 3h, missed connection, what compensation am I entitled to?" | billing_dispute (0.45) | 0.67 | **escalate** — high-risk intent | "I am sorry to hear about your delayed flight and missed connection. Please follow/DM your confirmation number so we can look into this for you. *AMV" |
| **AskPayPal** — "a charge I didn't make, I want it reversed and my account frozen" | cancellation (0.58) ❌ | 0.60 | **escalate** — high-risk intent | "I'm sorry to hear this. Please send us a DM with your PayPal email address so we can look into this charge and help secure your account. ^IN" |
| **AppleSupport** — "since the latest iOS update my battery drains in 3 hours, this is ridiculous" | complaint_feedback (0.52) ❌ | 0.80 | **escalate** — low confidence | "We'd like to look into this further with you. DM us the iOS version you are currently running and we'll go from there. `<link>`" |
| **comcastcares** — "when every other tweet ends with 'we apologize for the inconvenience' you know it's a terrible company" | complaint_feedback (0.92) | 0.85 | **auto_send** | "Hello, thanks for reaching out about your concerns and I'm happy to help! Can you DM the full address? ~AT" |

What this shows:
- **The RAG generator genuinely picks up brand voice from retrieval** — the
  Amazon "^EA" / Spotify "backstage /AL" / Delta "*AMV" sign-off styles are all
  learned from the retrieved exemplars, not hard-coded.
- **Routing is robust even when classification is wrong.** Three of six intents
  are wrong (Spotify, PayPal, Apple), but every one still escalates — twice via
  the high-risk-intent rule, once via low confidence. The rule engine is doing
  its job as a safety net over a weak classifier.
- **Failure mode #3 reproduces on real data:** the Amazon reply is flagged
  `grounded=False` for containing "tracking information" — a generic phrase, not
  an invented fact. The regex needs the normalisation fix.
- **Failure mode #2 reproduces on real data:** the Comcast message is sarcasm;
  the lexicon scores it as ordinary negative, intent is medium-risk, so it is the
  one message that auto-sends. Arguably it should escalate.

## 5. Why there are no real *quantitative* eval numbers yet

Two blockers, both external:

1. **No gold labels for the real data.** `make_labelset --unlabelled` produced
   `eval/labelset/labels.real.unlabelled.jsonl` (200 stratified real messages).
   Someone has to label `intent` + `should_escalate` by hand per
   [`labeling-guide.md`](labeling-guide.md) before `run_classification` /
   `run_escalation` can score anything.
2. **Gemini free-tier quota.** `gemini-3.x-flash` on the free tier allows **~20
   requests per day per model**. A full generation + judge + pairwise +
   zero-shot-baseline eval over 200 examples is ~800 calls. It needs either a
   paid key or several days of budgeted runs. The client already honours the
   server's `retryDelay` and has a `SUPPORT_AGENT_LLM_MIN_INTERVAL` throttle for
   when quota allows a slow run.

Everything except those two inputs is built and demonstrated working above.
