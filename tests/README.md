# Test Folder Guide

This folder is split by CI lane. Keep tests in the narrowest lane that can
protect the contract.

```text
tests/
├── README.md
├── __init__.py
├── utils.py
├── data/
├── docs/
│   ├── qwen3_omni/
│   └── s2pro/
├── test_model/
└── unit_test/
    ├── fixtures/
    ├── pipeline/
    ├── qwen3_omni/
    └── fishaudio_s2_pro/
```

## Root Files

- `README.md`: This file. It explains test ownership and where new tests belong.
- `__init__.py`: Keeps `tests` importable as a package.
- `utils.py`: Shared helpers used by docs and model CI tests.

Do not add root-level `test_*.py` files. Root test files are too easy for broad
pytest discovery to collect accidentally, and they make the CI lane unclear.

## `data/`

Small static fixtures shared by tests, such as images, audio, and short videos.
Keep these files small and deterministic. Large model artifacts, generated
outputs, and benchmark datasets should live outside the unit test tree.

## `docs/`

Documentation/example tests. These verify that documented user-facing examples
still work.

Use this lane when the test protects:

- install/docs snippets,
- client examples,
- documented request/response shapes,
- examples that may need optional docs dependencies.

These tests are not the default fast unit lane.

## `test_model/`

End-to-end and model CI tests. These are allowed to depend on real servers,
model snapshots, benchmark artifacts, optional packages, and GPU/runtime
resources.


## `unit_test/`


Expected command:

```bash
pytest tests/unit_test -q
```

Suggested ownership:

- `unit_test/pipeline/`: model-agnostic V1 pipeline contracts.
- `unit_test/qwen3_omni/`: Qwen3-Omni request/state/talker/audio contracts.
- `unit_test/fishaudio_s2_pro/`: FishAudio S2-Pro tokenizer/TTS/vocoder contracts.
- `unit_test/fixtures/`: small fake objects and payload factories.
