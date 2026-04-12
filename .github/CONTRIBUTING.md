# Contributing to MeaningBERT

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with GitHub

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [GitHub Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from the **`dev` branch**.
2. If you've added code that should be tested, you **must** ensure it is properly tested.
3. If you've changed APIs, update the documentation.
4. Ensure the CI/CD test suite passes.
5. Make sure your code lints.
6. Submit that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the
same [MIT License](https://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the
maintainers if that's a concern.

## Write bug reports with detail, background, and sample code

We use GitHub issues to track public bugs. Report a bug
by [opening a new issue](https://github.com/GRAAL-Research/MeaningBERT/issues). You should use one of
our [proposed templates](https://github.com/GRAAL-Research/MeaningBERT/tree/main/.github/ISSUE_TEMPLATE) when appropriate;
they are integrated with GitHub and do most of the formatting for you. It's that easy!

**Great Bug Reports** tend to have:

- A quick and clear summary and/or background
- Steps to reproduce
    - Be specific and clear!
    - Give sample code if you can. Try to reduce the bug to the minimum amount of code needed to reproduce: it will help
      in our troubleshooting procedure.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)
  Feel free to include any print screen or other file you feel may further clarify your point.

## Do you have a suggestion for an enhancement?

We use GitHub issues to track enhancement requests. Before you create an enhancement request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
  it first on the users list.

* Check the documentation to make sure your feature does not already exist.

* Do a [quick search](https://github.com/GRAAL-Research/MeaningBERT/issues) to see whether your enhancement has already
  been suggested.

When creating your enhancement request, please:

* Provide a clear title and description.

* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.

* Include code examples to demonstrate how the enhancement would be used.

## Prerequisites

To develop locally, you need to install the project dependencies. We use three requirements files:

- [styling_requirements.txt](https://github.com/GRAAL-Research/MeaningBERT/blob/main/styling_requirements.txt) for
  formatting and linting tools.
- [tests/requirements.txt](https://github.com/GRAAL-Research/MeaningBERT/blob/main/tests/requirements.txt) for testing
  dependencies.

You can install everything at once using the `all` extras:

```shell
pip install -e .[all]
```

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to enforce code quality checks before each commit. After installing the
dependencies, set up the hooks:

```shell
pre-commit install
```

This will automatically run formatting (Black), linting (PyLint), and various other checks on every commit. You can
also run the hooks manually on all files:

```shell
pre-commit run --all-files
```

## Use a Consistent Coding Style

All of the code is formatted using [black](https://black.readthedocs.io) with the
associated [config file](https://github.com/GRAAL-Research/MeaningBERT/blob/main/pyproject.toml). In order to format the
code of your submission, simply run

```shell
black .
```

Linting is done using [pylint](https://pylint.pycqa.org/) with the
associated [config file](https://github.com/GRAAL-Research/MeaningBERT/blob/main/.pylintrc):

```shell
pylint src/
```

## Running Tests

We use [pytest](https://docs.pytest.org/) for testing. To run the test suite:

```shell
pytest
```

This will automatically generate HTML and XML coverage reports as configured in
[pytest.ini](https://github.com/GRAAL-Research/MeaningBERT/blob/main/pytest.ini).

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References

This document was adapted from the open-source contribution guidelines
for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).
