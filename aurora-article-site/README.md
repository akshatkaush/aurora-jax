# Aurora Article GitHub Pages Site

This directory contains a static site for hosting the Medium article on Microsoft's Aurora in JAX, as requested.

## Local Preview

- You can preview `index.md` in VS Code using the Markdown preview (`Ctrl+Shift+V`).
- For a full GitHub Pages preview, install Ruby and Jekyll, then run:
  ```bash
  bundle init
  echo 'gem "github-pages", group: :jekyll_plugins' >> Gemfile
  bundle install
  bundle exec jekyll serve
  ```
- Visit `http://localhost:4000` in your browser.

## Publishing

Push this directory to a GitHub repository and enable GitHub Pages in the repo settings.