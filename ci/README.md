# CI workflow

`ci.yml` is the GitHub Actions workflow for this repo: on every push / PR it
installs the package, runs `ruff`, builds the synthetic corpus, trains the
classifier, and runs the full test suite + offline eval — all with the
deterministic offline backends (`SUPPORT_AGENT_EMBEDDER=hashing`,
`SUPPORT_AGENT_LLM=fake`), no secrets, no network.

It lives here instead of `.github/workflows/` only because the token used for the
initial push lacked GitHub's `workflow` OAuth scope. **To activate it:**

```bash
gh auth refresh -h github.com -s workflow   # one-time, approve in browser
git mv ci/ci.yml .github/workflows/ci.yml
git commit -m "Activate CI workflow" && git push
```
