# Development notes

## Adding a new weight measurement

Open `weights.csv` and append a new row in `DD/MM/YYYY,weight_kg` format.


## Regenerating the website

After saving `weights.csv`, run:

```bash
source .venv/bin/activate
python fit_model.py
```

This re-fits the logistic model, recomputes the confidence bands, and overwrites
`docs/index.html` with a fresh interactive Plotly chart. Then commit and push:

```bash
git add weights.csv docs/index.html
git commit -m "Add weight measurement YYYY-MM-DD"
git push
```

GitHub Pages will pick up the new `docs/index.html` automatically within a minute or two.

## First-time GitHub Pages setup

1. Push the repository to GitHub.
2. Go to **Settings → Pages**.
3. Set **Source** to `Deploy from a branch`, branch `main`, folder `/docs`.
4. Save. The site will be live at `https://<username>.github.io/<repo>/`.

## Automating with a GitHub Action (optional)

If you want the site to regenerate automatically whenever you push a new row to
`weights.csv` — without running `fit_model.py` locally — add this workflow file:

`.github/workflows/update-site.yml`

```yaml
name: Regenerate growth site

on:
  push:
    paths:
      - weights.csv   # only triggers when this file changes

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install numpy scipy pandas

      - name: Fit model and generate site
        run: python fit_model.py

      - name: Commit updated docs
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/index.html
          git diff --cached --quiet || git commit -m "chore: regenerate growth site [skip ci]"
          git push
```

With this in place, the full workflow is:

1. Edit `weights.csv` (e.g. directly on GitHub or via a push).
2. The Action runs `fit_model.py` and commits the new `docs/index.html`.
3. GitHub Pages deploys it automatically.

No local steps needed.
