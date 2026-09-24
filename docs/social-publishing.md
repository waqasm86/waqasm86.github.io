# Optional social publishing

The canonical article always remains at `https://waqasm86.github.io/`. Social tooling distributes a link; it is not part of the website runtime or Pages deployment.

```text
article merged
    -> GitHub Pages deploy
    -> LinkedIn text draft generated
    -> human review
    -> optional Buffer Idea or queue item
    -> LinkedIn
```

## Generate a LinkedIn draft

```bash
python -m tools.social.generate_linkedin \
  _posts/2026-09-24-why-i-built-kaggle-vllm.md
```

The output is written to `artifacts/social/<slug>/linkedin.txt`. The generator uses `verified_result` only when that field is explicitly present in article front matter; otherwise it describes methodology and says that no new quantitative result is introduced.

## Buffer dry run

Buffer is optional. The client follows Buffer's current GraphQL documentation: bearer authentication at `https://api.buffer.com`, `createPost` for a channel queue, and `createIdea` for an organization-level draft.

```bash
python -m tools.social.buffer_publish \
  artifacts/social/why-i-built-kaggle-vllm/linkedin.txt \
  --mode draft
```

Dry run is the default and makes no network call. It prints the planned operation without printing a token.

For a live **Idea** draft:

```bash
export BUFFER_API_KEY='set-locally-or-in-a-GitHub-secret'
export BUFFER_ORGANIZATION_ID='your-organization-id'
python -m tools.social.buffer_publish artifacts/social/<slug>/linkedin.txt \
  --mode draft --title 'Article draft' --execute
```

For the next queue slot:

```bash
export BUFFER_API_KEY='set-locally-or-in-a-GitHub-secret'
export BUFFER_CHANNEL_ID='your-linkedin-channel-id'
python -m tools.social.buffer_publish artifacts/social/<slug>/linkedin.txt \
  --mode queue --execute
```

Never write those values into Git, workflow YAML, an issue, or a build artifact. Live publishing is intentionally absent from CI.

## Retool and Substack

Retool is not a dependency. A future editorial system could show a GitHub-generated draft in a Retool approval dashboard, then call Buffer after approval.

The project does not use an unofficial Substack publishing API. Chirpy produces `/feed.xml`, which can support manual or future standards-based syndication while the canonical article stays on this site.

## API references

- [Buffer posts and scheduling](https://developers.buffer.com/guides/posts-and-scheduling.html)
- [Buffer Ideas](https://developers.buffer.com/guides/ideas.html)
