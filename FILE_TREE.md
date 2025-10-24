```plaintext
abx-next/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ .gitignore
├─ .pre-commit-config.yaml
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ docs/
│  ├─ index.md
│  └─ concepts/
│     ├─ variance_reduction.md
│     ├─ sequential.md
│     └─ switchback.md
├─ examples/
│  ├─ quickstart_ab.py
│  └─ power_simulation.py
├─ src/
│  └─ abx_next/
│     ├─ __init__.py
│     ├─ utils/
│     │  ├─ __init__.py
│     │  └─ types.py
│     ├─ analysis/
│     │  ├─ __init__.py
│     │  ├─ cuped.py
│     │  ├─ triggered.py
│     │  ├─ diff.py
│     │  └─ srm.py
│     └─ design/
│        ├─ __init__.py
│        └─ switchback.py
└─ tests/
   ├─ test_cuped.py
   ├─ test_srm.py
   ├─ test_triggered.py
   └─ test_power.py
```