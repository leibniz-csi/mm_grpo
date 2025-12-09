# Tests layout

Each folder under tests/ corresponds to a test category for a sub-namespace in gerl. For instance:
- `tests/workers` for testing functionality related to `gerl/workers`
- ...

There are a few folders with `special_` prefix, created for special purposes:
<!-- - `special_distributed`: unit tests that must run with multiple GPUs -->
- `special_docstring`: a suite of tests to check docstring examples
- `special_sanity`: a suite of quick sanity tests