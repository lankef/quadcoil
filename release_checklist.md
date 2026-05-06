# QUADCOIL PyPI Release Checklist

Use this checklist from the repository root.

## 1) Pre-release

- [ ] Confirm `version` in `pyproject.toml` is bumped and unique.
- [ ] Ensure working tree is clean or only contains intentional release edits.

## 2) Build and validate artifacts

- [ ] Install tools:

  ```bash
  python3 -m pip install --upgrade build twine
  ```

- [ ] Clean old artifacts:

  ```bash
  rm -rf dist build src/*.egg-info
  ```

- [ ] Build distributions:

  ```bash
  python3 -m build
  ```

- [ ] Validate distributions:

  ```bash
  python3 -m twine check dist/*
  ```

## 3) Record release hash

- [ ] Capture and record the main hash in docs/version history:

  ```bash
  git rev-parse main
  ```

## 4) Preview documentation locally

- [ ] Install the ReadTheDocs theme if needed:

  ```bash
  python3 -m pip install sphinx-rtd-theme
  ```

- [ ] Build the docs locally:

  ```bash
  python3 -m sphinx -b html docs docs/_build/html
  ```

- [ ] Open `docs/_build/html/index.html` in a browser and review the docs before sending them to ReadTheDocs.

- [ ] If `sphinx-rtd-theme` is unavailable, validate the RST with Sphinx's basic theme:

  ```bash
  python3 -m sphinx -b html -D html_theme=basic docs docs/_build/html-basic
  ```

- [ ] Open `docs/_build/html-basic/index.html` for the fallback preview. This validates most docs content, but it will not match the ReadTheDocs styling.

## 5) Upload to TestPyPI (recommended)

- [ ] Set TestPyPI token:

  ```bash
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="pypi-<your-testpypi-token>"
  ```

- [ ] Upload to TestPyPI:

  ```bash
  python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
  ```

## 6) Smoke test from TestPyPI

- [ ] Create fresh environment and install:

  ```bash
  python3 -m venv .venv-testpypi
  source .venv-testpypi/bin/activate
  python -m pip install --upgrade pip
  python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quadcoil==0.1.0
  python -c "import quadcoil; print('quadcoil import ok')"
  deactivate
  ```

## 7) Upload to production PyPI

- [ ] Set PyPI token:

  ```bash
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="pypi-<your-pypi-token>"
  ```

- [ ] Upload to PyPI:

  ```bash
  python3 -m twine upload dist/*
  ```

## 8) Finalize release

- [ ] Tag and push:

  ```bash
  git tag v0.1.0
  git push origin main --tags
  ```

- [ ] Verify package page:
  - https://pypi.org/project/quadcoil/
