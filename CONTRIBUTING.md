# Contributing

This repository publishes evidence-first technical writing. A pull request should make every quantitative statement traceable to an immutable source or a committed raw artifact.

## Content requirements

1. Start from a file in `templates/`.
2. Record the source repository and immutable commit SHA.
3. Keep code excerpts short and attributed; use `tools/article/extract_code.py` when practical.
4. Generate tables and charts from JSON or CSV evidence.
5. Separate observations, interpretations, failures, and limitations.
6. Do not include credentials, private logs, personal data, or unlicensed media.
7. Do not claim authorship, employment, performance, or benchmark outcomes without evidence.

## Validation

```bash
python -m tools.article.validate_frontmatter
python -m tools.article.check_links
bundle exec ruby tools/article/check_rouge.rb
python -m pytest
bundle exec jekyll build
python -m tools.article.check_generated_site
bundle exec htmlproofer _site --disable-external
```

External links are reviewed by a human because transient network failures should not block every content pull request.
