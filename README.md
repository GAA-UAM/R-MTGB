#  Robust Multi-Task Gradient Boosting (R-MTGB)

A robust and scalable multi-task learning (MTL) framework that integrates outlier task detection into a structured gradient boosting process. Built with Python and [scikit-learn](https://scikit-learn.org/), R-MTGB is designed to generalize well across heterogeneous task sets and is resilient to task-level noise.

---

## 📘 About

**R-MTGB** (Robust Multi-Task Gradient Boosting) is a novel ensemble-based learning framework developed to handle task heterogeneity and task-level noise in multi-task learning settings. The model introduces a three-stage boosting architecture:

1. **Shared Representation Learning:** Learns features common across all tasks.
2. **Outlier Task Detection & Weighting:** Optimizes regularized, task-specific parameters to dynamically down-weight noisy or outlier tasks.
3. **Task-Specific Fine-Tuning:** Refines models individually to capture task-specific nuances.

---

## ✨ Features

- Multi-task learning with task-specific and shared components.
- Automatic outlier task detection.
- Gradient boosting-based architecture with interpretability.
- Compatible with various loss functions (regression/classification).
- Performance analysis with per-task metrics.
- Synthetic data generator for benchmarking.
- Scikit-learn compatible design.

---

## 💻 Installation

Clone the repository and install dependencies using [requirements](/requirements.txt)

```bash
git clone https://github.com/GAA-UAM/R-MTGB.git
cd R-MTGB
pip install -r requirements.txt
```

---

## License
The package is licensed under the GNU Lesser General Public [License v2.1](LICENSE).

## 📚 Citations & References
If you use R-MTGB in your research or work, please consider citing this project using the corresponding citation format:
```yml

```

## 👨‍💻 Authors
- [Seyedsaman (Saman) Emami](https://github.com/samanemami/)
- [Gonzalo Martínez-Muñoz](https://github.com/gmarmu)
- [Daniel Hernández-Lobato](https://github.com/danielhernandezlobato)

---

## Documentation 
To get started with this project, please refer to the [Wiki](https://github.com/GAA-UAM/R-MTGB/wiki)."

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

---

## 💾 Release Information

### Version
0.0.1

### Updated
05 June 2025

### Date-released
26 Jan 2024
