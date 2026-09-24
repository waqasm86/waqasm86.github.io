# Publishing architecture

The publication pipeline treats code and experiment artifacts as evidence. It never turns a Git diff directly into unreviewed prose.

```text
kaggle-vllm article-ready commit/release
    |
    | workflow_dispatch (repository_dispatch may be added later)
    v
waqasm86.github.io import workflow
    |
    +-- fetch public source at immutable SHA
    +-- read a reviewed publishing manifest
    +-- extract selected committed lines
    +-- ingest named JSON/CSV evidence
    +-- render review artifacts
    v
human-reviewed publication pull request
    |
    v
merge -> GitHub Pages build -> deploy
```

The current `import-kaggle-vllm.yml` workflow stops at a downloadable Actions artifact. It does not generate prose, commit changes, open a pull request, or publish.

## Future cross-repository dispatch

A source repository could emit `repository_dispatch` after an article-ready PR or release. Triggering a workflow in another repository may require a narrowly scoped fine-grained personal access token or a GitHub App, depending on repository ownership and authentication policy. A future implementation should grant access only to the target repository and required workflow operation. A broad classic token is not appropriate.

Reading the public source repository does not require a PAT. No dispatch credential is configured in this repository.

## Evidence rules

- Fetch source by full commit SHA or an immutable tag resolved and recorded as a SHA.
- Refuse dirty source files during extraction.
- Preserve raw evidence and checksums.
- Generate charts and tables mechanically.
- Require a person to review the claim boundary before merge.
