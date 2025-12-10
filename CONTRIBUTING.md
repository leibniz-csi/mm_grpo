# Contributing to MM-GRPO

We welcome contributions of any kind of bug fixes, enhancements, documentation improvements, or even just feedback.

Your support can take many forms:
- Report issues or unexpected behaviors in [Issues](https://github.com/leibniz-csi/mm_grpo/issues).
- Suggest or implement new features in [Issues](https://github.com/leibniz-csi/mm_grpo/issues).
- Add or improve functions, or expand documentation by submitting a pull request, refer to [Getting Started](#getting-started) for more guidelines.
- Review [pull requests](https://github.com/leibniz-csi/mm_grpo/pulls) and assist other contributors.
- Share mm_grpo in blog posts, social media, or give the repo a ⭐.

## Finding Issues to Contribute

Looking for ways to dive in? You can learn the development plan and roadmap via [Roadmap](https://github.com/leibniz-csi/mm_grpo/issues?q=is%3Aissue%20state%3Aopen%20label%3Aroadmap).


## Getting Started
If you want to submit a pull request to our repository, here are steps to set up `mm_grpo` for local development and contribution.

1. Fork `mm_grpo` repo on [Github](https://github.com/leibniz-csi/mm_grpo).

2. Clone your fork locally:
  ```bash
  git clone git@github.com:your_name_here/mm_grpo.git
  ```
  After that, add the official repository as the upstream repository:
  ```bash
  git remote add upstream git@github.com:leibniz-csi/mm_grpo.git
  ```

3. Install your local copy into your local environment. You can refer to [Installation](README.md#installation).

4. Create a branch for local development
  ```bash
  git checkout -b name-of-your-bugfix-or-feature
  ```
  Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the linters and the tests:
  ```bash
  # Assume you have installed `pre-commit`
  pre-commit run --show-diff-on-failure --color=always --all-files
  ```

6. Commit your changes and push your branch to GitHub:
  ```bash
  git add .
  git commit -m "Your detailed description of your changes."
  git push origin name-of-your-bugfix-or-feature
  ```

7. Submit a pull request through the GitHub website.

## Testing

Currently, we use local unit tests to ensure code quality.
To install the testing dependencies, run:

```bash
pip install pytest pytest-cov pytest-asyncio
```

To run the tests, execute:

```bash
python3 -m pytest --cov=gerl/workers tests/workers
```

If you make changes to the codebase, please ensure that you add or update the corresponding tests to cover your modifications.


## Pull Requests Guidelines

Thanks for submitting a PR! To streamline reviews:
- Adhere to our pre-commit lint rules and ensure all checks pass.
- Update docs for any user-facing changes or new functionalities.
- Add or update tests if the pull request adds functionality.
<!-- - Follow our Pull Request Template for title format and checklist. -->

## License

See the [LICENSE](https://github.com/leibniz-csi/mm_grpo/blob/main/LICENSE) file for full details.
