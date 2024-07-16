# Genetic Marker Classification for Breast Cancer

## Project Overview
This project focuses on the identification of genetic markers associated with breast cancer using machine learning-based classification. The goal is to develop and evaluate models that can accurately classify genetic data to identify potential markers.

## Directory Structure

```
/genomics
│
├── data
│ ├── raw
│ │ └── *.csv (initial genome.ucsc.edu and favor.genohub.org data)
│ ├── processed
│ │ └── *.csv (processed data files)
│ ├── external
│ │ └── *.csv (external datasets)
│ └── interim
│ └── *.csv (intermediate datasets)
│
├── models
│ ├── trained_models
│ │ └── *.pkl (trained model files)
│ └── model_checkpoints
│ └── *.ckpt (checkpoint files)
│
├── scripts
│ ├── data_processing
│ │ └── *.py (scripts for data cleaning and processing)
│ ├── model_training
│ │ └── *.py (scripts for training models)
│ └── evaluation
│ └── *.py (scripts for model evaluation)
│
├── notebooks
│ └── *.ipynb (Jupyter notebooks for exploration and analysis)
│
├── results
│ ├── figures
│ │ └── *.png, *.pdf (plots and figures)
│ ├── logs
│ │ └── *.log (log files)
│ └── reports
│ └── *.txt, *.md (summary reports and notes)
│
├── docs
│ └── *.md (documentation files)
│
├── README.md (project overview and instructions)
└── requirements.txt (dependencies for the project)
```

## Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Data
- **raw:** Contains raw data files in CSV format.
- **processed:** Contains processed data files in CSV format.
- **external:** Contains external datasets used in the project.
- **interim:** Contains intermediate datasets generated during data processing.

## Models
- **trained_models:** Contains the saved trained models in PKL format.
- **model_checkpoints:** Contains checkpoint files from model training.

## Scripts
- **data_processing:** Contains scripts for data cleaning and preprocessing.
- **model_training:** Contains scripts for training machine learning models.
- **evaluation:** Contains scripts for evaluating the performance of the models.

## Notebooks
Contains Jupyter notebooks used for exploration, analysis, and visualization.

## Results
- **figures:** Contains generated plots and figures.
- **logs:** Contains log files from model training or other processes.
- **reports:** Contains summary reports and notes.

## Documentation
Contains detailed descriptions of methods, data sources, and assumptions.

## Usage
- **Data Preprocessing:** Run the scripts in the data_processing folder to clean and preprocess the data.
- **Model Training:** Use the scripts in the model_training folder to train your models.
- **Evaluation:** Evaluate the models using the scripts in the evaluation folder.
- **Exploration and Analysis:** Use the Jupyter notebooks in the notebooks folder for further analysis and visualization.

## Contributing
If you wish to contribute to this project, please follow these steps:
- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Commit your changes (`git commit -m 'Add new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or inquiries, please contact [Camilo Pérez] at [camiloperez@javerianacali.edu.co].
