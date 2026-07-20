# ETHOS Test Suite

This directory contains the production-ready pytest suite for the ETHOS simulation.

## Running Tests

From the project root, run:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_engines.py
```

## Fixtures

Shared test fixtures (mock citizens, agents, streets, and cities) are defined in
`tests/conftest.py` and are automatically discovered by pytest.
