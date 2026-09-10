# Non-obvious decisions

Thirteen choices a reasonable engineer could have made differently, and why this
project made them the way it did. Each is phrased so it can be argued with.

---

### 0. Focus brand: AmazonHelp, but a cross-brand classifier

The brief says pick one brand. **AmazonHelp** was chosen over AppleSupport (the
other 4k-thread brand) because its inbound mix actually spans the taxonomy —
delivery, order status, refunds, account access, Prime cancellation, Alexa/app
bugs, product questions — whereas Apple's is ~90% device bugs and would make the
classification story trivial. The retrieval corpus, brand voice, evaluation set,
and escalation tuning are all AmazonHelp-only. The **classifier** is the one
exception: 4k Amazon threads weak-label to too few rows per class
(`product_question` = 1, `technical_bug` = 11) to train on, so it is trained on
the cross-brand corpus and *deployed* for Amazon. The intent taxonomy is still
defined from AmazonHelp's own clusters. A reviewer could argue for an
Amazon-only classifier with hand-labelled training data; that is roadmap item 1.

### 1. Logistic regression on frozen embeddings, not a fine-tuned transformer

A fine-tuned DistilBERT would very likely score a few macro-F1 points higher.
It was rejected because: (a) the assignment weights *proof* over system quality
and a linear model's coefficients are directly inspectable — for any prediction
I can list the embedding dimensions that pushed it; (b) it trains in seconds on
a laptop CPU with no GPU and no API key; (c) it degrades gracefully — the same
code runs with the hashing embedder for anyone who will not download a model.
The cost is real and stated: on `order_status` the linear model's recall is
weak (see report), and a transformer would probably fix that.

### 2. Weak supervision for training labels; hand-labels spent only on eval

The dataset has no intent labels and hand-labelling millions of tweets is
infeasible. Keyword/regex labelling functions (`classify/weak_label.py`) produce
a noisy training set; humans label only the 200-example evaluation set. This is
the standard Snorkel-style trade and it has a specific failure mode we accept:
the classifier partly learns the labelling rules' blind spots. That is exactly
why the headline metric is macro-F1 on the *hand-labelled* set, never CV
accuracy on weak labels (which is ~0.99 and almost meaningless — it mostly says
the embeddings linearly separate the rules' own decision boundary).

### 3. Intents discovered by clustering, then frozen by a human — not chosen by an LLM

`taxonomy/discover.py` embeds a sample, runs K-Means across a range of k, picks k
by silhouette, and prints top terms + nearest messages per cluster. A person
names and merges those into the 10 intents in `taxonomy/intents.py`. Asking an
LLM to "propose intents" is faster but gives a taxonomy nobody can defend the
boundaries of. This way the taxonomy is grounded in the data's actual structure
and the merge decisions are recorded.

### 4. Ten intents, with a deliberate `other_unclear` bucket

Fewer than ~8 and distinct problems (billing vs cancellation) collapse together;
more than ~12 and per-class eval support gets too thin to trust. `other_unclear`
is not a failure class — it is a first-class routing target for messages too
short or ambiguous to action, and it keeps the other nine classes clean.

### 5. Every intent carries a risk tier; escalation keys off it

`billing_dispute`, `cancellation`, `account_access` are `high` risk: a wrong
auto-reply there moves money or loses an account. Those escalate on intent alone,
*regardless of classifier confidence*. This encodes a product judgment — "some
mistakes are not worth any automation upside" — directly in the taxonomy rather
than hoping a confidence threshold catches them.

### 6. Escalation is an ordered rule list, not a learned classifier or a prompt

A second ML model for "should this escalate" would need its own labelled set and
would be just as opaque as the thing it guards. An LLM prompt ("should a human
see this?") is non-deterministic and unauditable. The rule engine
(`escalate/rules.py`) returns the winning rule *and every other rule's truth
value*, so a reviewer sees the whole decision surface. Order matters and is
justified in the file: irreversible/regulated first, then "model unsure", then
soft signals.

This paid off on the real-data eval. The first run auto-sent 5 escalation-worthy
messages; because every signal is inspectable we could read the 5, see that the
lexicon sentiment scored "you all stole my package" as neutral, and add three
targeted rules (`severity_cue`, `high_risk_intent_suspected`,
`negative_sentiment_routine_intent`) that took `false_auto_send` to 0 without
touching the rest. A learned escalation model would have needed retraining and
offered no such handle. See `docs/real-data-notes.md` §5.

### 7. Cost-weighted escalation metric: a false auto-send costs 5× a false escalation

Plain accuracy or F1 treats both errors equally. They are not equal: auto-sending
a wrong reply to an angry customer with a billing dispute is a much worse outcome
than routing a routine question to a human who rubber-stamps it. The 5:1 ratio is
a documented assumption in `config.py`, swept in the report, and is the number
the rule engine is actually tuned against.

### 8. Retrieval is filtered to the *same brand* and to *resolved* threads

Brand voice is real — "^JD, BoltMobile Care" vs "— The Nimbus Air Team ✈️" — so
exemplars come from the same brand where possible; a thin same-brand result
(< `k_min`) is itself an escalation signal. "Resolved" is inferred structurally
(agent had the last turn, customer did not re-open). Both are approximations with
stated costs: the resolved proxy counts "agent gave up politely" as success, and
cross-brand fallback dilutes voice.

### 9. `LLMClient` is a 1-method protocol with a deterministic Fake

The entire pipeline depends only on `complete(prompt, *, system, temperature,
json_mode)`. Gemini, the Fake, or any future provider slot in without touching a
line of pipeline code. The Fake is not a mock that returns constants — it
dispatches on the prompt (classification → JSON label, generation → templated
reply from the exemplar, judge → rubric JSON) so tests exercise real control
flow. This is what makes CI possible with no secrets.

### 10. A hashing embedder as a first-class fallback, not just a test stub

`SUPPORT_AGENT_EMBEDDER=hashing` gives a deterministic 512-d char-n-gram vector,
zero download, byte-identical across machines. CI uses it; so can any reviewer
who does not want a 90 MB model pull. It is measurably worse (the report shows
both configs) but it keeps the whole system — including the escalation result,
which does not depend on embedding quality much — runnable and reproducible
anywhere.

### 11. Groundedness is checked by regex after generation, and feeds escalation

After the LLM drafts a reply, `generator._check_grounded` scans for concrete
specifics — order numbers, dates, dollar amounts, times — that appear in the
draft but not in the customer's message. A hit sets `grounded=False`, which the
`ungrounded_draft` rule turns into an escalation. It is a cheap, transparent
hallucination guard that does not need a second model call. It over-triggers on
legitimately quoted-back numbers; that is the safe direction.

### 12. Config is a frozen dataclass tree, and thresholds are tuned on a dev split

Every magic number (confidence 0.55, margin 0.15, weak-similarity 0.45, the 5:1
cost) lives in `config.py`. The escalation thresholds are selected by
`eval/tune_thresholds.py` on a 30% *dev* slice of the label set and only then
pasted back with a `# tuned` comment; the headline eval runs on the untouched
test slice. This keeps "we picked the number that made the graph look good" out
of the reported figures.

---

### Rejected alternatives, briefly

- **End-to-end LLM agent** (one prompt does classify + reply + route): highest
  ceiling, but a single failure is unattributable and there is no baseline to
  compare against. The whole point here was attributability.
- **Vector DB (FAISS/Chroma)** for retrieval: unnecessary at this corpus size;
  a NumPy matrix multiply is faster to reason about and has no service to run.
- **Fine-tuned reply model**: no clean training target (historical replies are
  noisy and often reference context we do not have), and it would remove the
  retrieval-only baseline that currently falls out for free.
- **Per-brand classifiers**: too little data per brand; one shared classifier
  with brand handled at retrieval time is the better split.
