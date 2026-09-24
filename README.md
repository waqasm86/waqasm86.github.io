# Mohammad Waqas — AI Inference & GPU Systems

Source for [waqasm86.github.io](https://waqasm86.github.io/): a long-form technical publication and engineering notebook about vLLM, CUDA, GPU systems, distributed inference, Kubernetes, observability, and reproducible AI infrastructure experiments.

This is not a conventional portfolio. Articles connect engineering questions to source code, environment identity, experiment design, raw evidence, negative results, and limitations.

## Architecture

- Jekyll with `jekyll-theme-chirpy` 7.6
- Markdown and Rouge syntax highlighting with block line numbers
- theme-native MathJax and Mermaid loading through post front matter
- GitHub Actions and GitHub Pages artifact deployment
- Python 3.11-compatible evidence, chart, validation, and social-draft tools
- no database, server process, React application, or paid hosting dependency

## Local development

The CI and Pages build use Ruby 3.3. Install a compatible Ruby and Bundler, then:

```bash
git clone https://github.com/waqasm86/waqasm86.github.io.git
cd waqasm86.github.io
bundle install
bundle exec jekyll serve --livereload
```

Open `http://127.0.0.1:4000/`.

For publication tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Content structure

```text
_posts/                  published technical articles
_tabs/                   Projects, Research, Series, and About pages
assets/img/posts/        per-article images
assets/data/             raw public article evidence
templates/               article and social templates
publishing/              declarative import manifests
tools/article/           extraction and content validation
tools/render/            deterministic charts and social cards
tools/social/            LinkedIn drafts and optional Buffer client
```

## Create the next article

1. Copy the appropriate template to a dated post filename:

   ```bash
   cp templates/article.md _posts/2026-10-01-measuring-tensor-parallelism.md
   mkdir -p assets/img/posts/measuring-tensor-parallelism
   mkdir -p assets/data/measuring-tensor-parallelism
   ```

2. Replace every placeholder and record immutable source/evidence identities.
3. Generate the social preview:

   ```bash
   python -m tools.render.social_card \
     --title 'Measuring Tensor Parallelism on Dual NVIDIA T4 GPUs' \
     --series 'kaggle-vllm Engineering Notes' \
     --output assets/img/posts/measuring-tensor-parallelism/cover.png
   ```

4. Generate charts from committed evidence, never from invented or hand-entered values:

   ```bash
   python -m tools.render.benchmark_chart \
     --input assets/data/measuring-tensor-parallelism/results.csv \
     --x concurrency --y throughput \
     --title 'Concurrency vs throughput' \
     --x-label 'Concurrent requests' --y-label 'Output tokens/s' \
     --output assets/img/posts/measuring-tensor-parallelism/benchmark-throughput.png
   ```

5. Run the full validation commands below and open a pull request.

## Extract committed code from kaggle-vllm

Check out the source at the exact commit required by the article:

```bash
git clone https://github.com/kaggle-vllm/kaggle-vllm.git ../kaggle-vllm-public
git -C ../kaggle-vllm-public checkout <full-commit-sha>
python -m tools.article.extract_code \
  --repo ../kaggle-vllm-public \
  --file examples/dual_t4_tp2.py \
  --lines 3:16 \
  --language python \
  --source-url 'https://github.com/kaggle-vllm/kaggle-vllm/blob/<sha>/examples/dual_t4_tp2.py'
```

The extractor refuses uncommitted source content and records the file, commit, line range, and immutable link.

To process a reviewed manifest:

```bash
python -m tools.article.render_manifest \
  --manifest publishing/example.yml \
  --repo ../kaggle-vllm-public \
  --output-dir artifacts/import
```

The GitHub import workflow performs the same process and uploads review artifacts. It does not publish prose or commit changes.

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

The content workflow runs those checks for pull requests and pushes to `main`. The Pages workflow rebuilds and deploys only after its build and HTML checks pass.

## Deployment

`.github/workflows/pages-deploy.yml` uses the official Pages artifact flow with minimal permissions: `contents: read`, `pages: write`, and `id-token: write`. In repository settings, choose **GitHub Actions** as the Pages source. Do not configure a `gh-pages` branch.

## Social drafts

```bash
python -m tools.social.generate_linkedin _posts/<dated-slug>.md
python -m tools.social.buffer_publish artifacts/social/<slug>/linkedin.txt --mode draft
```

The second command is a dry run unless `--execute` is supplied. Buffer is optional and never required to build or deploy the site. See [docs/social-publishing.md](docs/social-publishing.md).

## Security and evidence policy

- Never commit PATs, Buffer keys, LinkedIn tokens, Retool tokens, or publishing credentials.
- Never publish a benchmark number without a named source artifact.
- Never infer authorship from repository presence alone.
- Prefer immutable commit, release, checksum, JSON/CSV, log, PR, and workflow-run links.
- Keep the source `kaggle-vllm` repository independent; this site only reads public revisions.

## License

Site code and original publication material in this repository are available under the [MIT License](LICENSE). Linked source projects and quoted excerpts retain their own licenses.
