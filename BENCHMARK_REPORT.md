# Respiratory Rate Prediction & Improvement Benchmark Report

**Generated:** 2026-09-27 23:55:05

This document tracks the incremental progress and accuracy benchmarks across every optimization phase.

---

## 1. Incremental Progress Summary (Ensemble Model)

| Step ID | Step Name | MAE (bpm) | RMSE | R² Score | F1 Macro | Within ±1.5 bpm | Within ±2.0 bpm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **step_0** | Baseline (Fixed ROI Fallback) | **1.921** | 2.246 | -0.347 | **0.458** | 40.6% | 59.4% |
| **step_1** | MediaPipe Dynamic Tracking | **1.92** | 2.245 | -0.346 | **0.458** | 40.6% | 59.4% |
| **step_live** | Live Terminal Showcase Training | **1.92** | 2.245 | -0.346 | **0.458** | 40.6% | 59.4% |

---

## 2. Detailed Step-by-Step Benchmarks

### Phase: Baseline (Fixed ROI Fallback) (`step_0`)
- **Timestamp**: 2026-09-27T23:49:56.327606
- **Description**: Initial system without MediaPipe installed; using fixed rectangular fallback box on thermal video.

#### Model Performance Comparison Table:

| Model | MAE | RMSE | R² | F1 Macro | F1 Weighted | Accuracy | Within ±1.5 bpm | Within ±2.0 bpm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 1.804 | 2.132 | -0.214 | **0.458** | 0.829 | 0.844 | 50.0% | 62.5% |
| **Gradient Boosting** | 2.063 | 2.406 | -0.546 | **0.429** | 0.777 | 0.75 | 40.6% | 53.1% |
| **XGBoost** | 2.106 | 2.542 | -0.726 | **0.407** | 0.738 | 0.688 | 46.9% | 53.1% |
| **Ensemble** | 1.921 | 2.246 | -0.347 | **0.458** | 0.829 | 0.844 | 40.6% | 59.4% |

#### Confusion Matrices:

**Random Forest (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 27 | 2 |
| **Tachypnea** | 3 | 0 |

![Random Forest Confusion Matrix](result/confusion_matrix_step_0_random_forest.png)

**Gradient Boosting (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 24 | 5 |
| **Tachypnea** | 3 | 0 |

![Gradient Boosting Confusion Matrix](result/confusion_matrix_step_0_gradient_boosting.png)

**XGBoost (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 22 | 7 |
| **Tachypnea** | 3 | 0 |

![XGBoost Confusion Matrix](result/confusion_matrix_step_0_xgboost.png)

**Ensemble (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 27 | 2 |
| **Tachypnea** | 3 | 0 |

![Ensemble Confusion Matrix](result/confusion_matrix_step_0_ensemble.png)

---

### Phase: MediaPipe Dynamic Tracking (`step_1`)
- **Timestamp**: 2026-09-27T23:54:12.586438
- **Description**: Feature extraction with dynamic MediaPipe Face Mesh nostril tracking.

#### Model Performance Comparison Table:

| Model | MAE | RMSE | R² | F1 Macro | F1 Weighted | Accuracy | Within ±1.5 bpm | Within ±2.0 bpm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 1.803 | 2.132 | -0.214 | **0.458** | 0.829 | 0.844 | 50.0% | 62.5% |
| **Gradient Boosting** | 2.063 | 2.406 | -0.546 | **0.429** | 0.777 | 0.75 | 40.6% | 53.1% |
| **XGBoost** | 2.106 | 2.542 | -0.726 | **0.407** | 0.738 | 0.688 | 46.9% | 53.1% |
| **Ensemble** | 1.92 | 2.245 | -0.346 | **0.458** | 0.829 | 0.844 | 40.6% | 59.4% |

#### Confusion Matrices:

**Random Forest (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 27 | 2 |
| **Tachypnea** | 3 | 0 |

![Random Forest Confusion Matrix](result/confusion_matrix_step_1_random_forest.png)

**Gradient Boosting (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 24 | 5 |
| **Tachypnea** | 3 | 0 |

![Gradient Boosting Confusion Matrix](result/confusion_matrix_step_1_gradient_boosting.png)

**XGBoost (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 22 | 7 |
| **Tachypnea** | 3 | 0 |

![XGBoost Confusion Matrix](result/confusion_matrix_step_1_xgboost.png)

**Ensemble (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 27 | 2 |
| **Tachypnea** | 3 | 0 |

![Ensemble Confusion Matrix](result/confusion_matrix_step_1_ensemble.png)

---

### Phase: Live Terminal Showcase Training (`step_live`)
- **Timestamp**: 2026-09-27T23:55:05.844920
- **Description**: Live interactive training pipeline execution with dynamic MediaPipe tracking.

#### Model Performance Comparison Table:

| Model | MAE | RMSE | R² | F1 Macro | F1 Weighted | Accuracy | Within ±1.5 bpm | Within ±2.0 bpm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 1.803 | 2.132 | -0.214 | **0.458** | 0.829 | 0.844 | 50.0% | 62.5% |
| **Gradient Boosting** | 2.063 | 2.406 | -0.546 | **0.429** | 0.777 | 0.75 | 40.6% | 53.1% |
| **XGBoost** | 2.106 | 2.542 | -0.726 | **0.407** | 0.738 | 0.688 | 46.9% | 53.1% |
| **Ensemble** | 1.92 | 2.245 | -0.346 | **0.458** | 0.829 | 0.844 | 40.6% | 59.4% |

#### Confusion Matrices:

**Random Forest (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 27 | 2 |
| **Tachypnea** | 3 | 0 |

![Random Forest Confusion Matrix](result/confusion_matrix_step_live_random_forest.png)

**Gradient Boosting (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 24 | 5 |
| **Tachypnea** | 3 | 0 |

![Gradient Boosting Confusion Matrix](result/confusion_matrix_step_live_gradient_boosting.png)

**XGBoost (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 22 | 7 |
| **Tachypnea** | 3 | 0 |

![XGBoost Confusion Matrix](result/confusion_matrix_step_live_xgboost.png)

**Ensemble (Labels: Normal, Tachypnea)**:

| True \ Pred | Normal | Tachypnea |
| :--- | :---: | :---: |
| **Normal** | 27 | 2 |
| **Tachypnea** | 3 | 0 |

![Ensemble Confusion Matrix](result/confusion_matrix_step_live_ensemble.png)

---
