# Noise Aware Boost

Multi-Task Noise-Aware Gradient Boosting: A Multi-Task ensemble for Enhanced Data Analysis with Robust Outlier Handling.

## Table of Contents

- [Introduction](#Introduction)
- [Features](#Features)
- [Usage](#Usage)
- [Dependencies](#Dependencies)
- [License](#License)
- [Key members](#Key_members)
- [Contributing](#contributing)
- [Version](#Version)

## Introduction

The Noise-Aware Gradient Boosting model is developed to provide a robust solution for data analysis tasks, addressing the challenges posed by outliers and noisy data. The dual-task nature of the model allows it to simultaneously focus on enhancing predictive performance and handling outliers effectively.

## Features

- Multi-task optimization for enhanced data analysis.
- Robust outlier handling to improve model resilience.
- Easy integration into various Gradient Boosting frameworks.
- Compatibility with popular machine learning frameworks.

## Usage

To use the model in your project, follow the following steps:

```Python
# For classification

from NoiseAwareGB.model import *

model = Classifier(
            max_depth=5,
            n_estimators=100,
            subsample=0.5,
            max_features="sqrt",
            learning_rate=0.05,
            random_state=1,
            criterion="squared_error",
        )


model.fit(x_train, y_train, task_train)
pred = model_mt.predict(x_test, task_test)

```

For more advanced usage and customization options, refer to the wiki page.

## Dependencies

Make sure you have the dependencies ([requirements](requirements.txt)) installed before using the Noise-Aware Gradient Boosting model.

```bash
pip install -r requirements.txt
```

## License

The package is licensed under the [GNU Lesser General Public License v2.1](LICENSE).

## Key_members

- [Seyedsaman Emami](https://github.com/samanemami)
- [Gonzalo Martínez-Muñoz](https://github.com/gmarmu)
- [Daniel Hernandez Lobato](https://github.com/danielhernandezlobato)

## Contributing

We welcome contributions from the community. If you find a bug, have a feature request, or want to contribute to the documentation, please follow our Contribution Guidelines.

## Version

0.0.1

## Date-released

20-Feb-2024

## Updated

20-Feb-2024
