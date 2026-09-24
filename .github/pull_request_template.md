## Purpose

## Evidence

- Source repository:
- Commit SHA or release:
- Raw JSON/CSV/logs:

## Claim boundary

- [ ] Quantitative statements are generated from or linked to source artifacts.
- [ ] Observations and interpretations are clearly separated.
- [ ] Failures and limitations are included.
- [ ] No credentials, private logs, or unrelated generated files are present.

## Validation

- [ ] `python -m tools.article.validate_frontmatter`
- [ ] `python -m tools.article.check_links`
- [ ] `python -m pytest`
- [ ] `bundle exec jekyll build`
- [ ] `bundle exec htmlproofer _site --disable-external`
