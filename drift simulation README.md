# Drift Simulator — MLOps Stress-Testing Framework

A modular Python framework designed to simulate and evaluate Data Drift, Target Drift, and Concept Drift against a production-trained machine learning baseline. 

It provides reproducible stress-testing pipelines using fixed baseline bins, schema validation, and configurable drift scenarios to ensure models meet strict SLA limits before deployment.

---

## 🏗️ Repository Architecture

```plaintext
drift_simulator/
├── config/                         # Configuration & Data Schemas
│   ├── baseline_config.yml          # Core drift thresholds, seeds, & pipeline parameters
│   ├── diabetes_schema.yaml         # Raw dataset schema validation rules
│   ├── diabetes_cleaned.yaml        # Cleaned dataset specifications
│   ├── diabetes_features.yaml       # Feature engineering schema definitions
│   ├── performance.yaml             # Model performance baseline & feature alignment
│   ├── X_test.yaml                  # Feature test distribution schemas
│   └── Y_test.yaml                  # Target test distribution schemas
│
├── baseline/                       # Baseline Computing & Schema Engine
│   ├── baseline_calculator.py       # Computes fixed PSI bin edges & statistical baselines
│   ├── schema_validator.py         # Validates data batches against config schemas
│   └── .cache/                     # Local disk cache for computed baselines
│       └── baseline_stats_v2.json  # Cached baseline stats & fixed bin grids
│
├── drift/                          # Drift Simulation Engines
│   ├── base_drift_simulator.py     # Base class (OOP) handling pipelines & PSI metrics
│   ├── no_drift/                   # Control group simulator
│   │   └── no_drift_simulator.py
│   ├── normal_drift/               # Moderate / Expected drift simulator
│   │   └── normal_drift_simulator.py
│   └── severe_drift/               # Extreme / Stress-test drift simulator
│       └── severe_drift_simulator.py
│
├── run_drift.py                    # Entry point CLI script to execute simulations
├── requirements.txt                # Third-party dependencies
└── *_results.json                  # Output evaluation reports (No Drift, Normal, Severe)
