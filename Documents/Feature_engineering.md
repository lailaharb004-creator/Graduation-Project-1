# Feature Engineering

## Introduction

Feature engineering is one of the most important components of the project.

The system creates additional features from raw GPS measurements to improve spoofing detection accuracy.

---

# Satellite Ratio

```python
sat_ratio = sat_locks / sat_count
```

Measures the percentage of visible satellites that are successfully locked.

---

# Satellite Discrepancy

```python
sat_discrepancy = abs(
    sat_count - sat_locks
)
```

Measures inconsistency between visible and locked satellites.

Large discrepancies may indicate GPS spoofing.

---

# Velocity Difference

```python
velocity_diff
```

Measures sudden speed changes.

Spoofed GPS signals often produce unrealistic velocity jumps.

---

# Heading Difference

```python
heading_abs_diff
```

Measures the difference between:

- Vehicle Heading
- GPS Course

Large deviations may indicate manipulated GPS information.

---

# Rolling Statistics

Window Size:

```python
window = 5
```

Generated Features:

```text
velocity_mean_5
velocity_std_5

sat_count_mean_5
sat_count_std_5

sat_ratio_mean_5
sat_ratio_std_5
```

Purpose:

- Capture temporal behavior
- Improve pattern recognition
- Reduce noise