# Solar Power Plants Dataset - California 🌞

![Solar Power Plants](https://img.shields.io/badge/Dataset-Solar%20Power%20Plants-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Version](https://img.shields.io/badge/Version-1.0.0-yellow)

Welcome to the **Solar Power Plants Dataset - California** repository!

---
## Table of Contents

0. [Project Organization](#project-organization)
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
   - [Columns](#columns)
   - [Key Features](#key-features)
   - [Target Variable](#target-variable)
   - [Data Quality Notes](#data-quality-notes)
3. [Use Cases](#use-cases)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Quality Notes](#data-quality-notes)
7. [Acknowledgments](#acknowledgments)
8. [Contact](#contact)

---

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data used for this project
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.Naming convention is a number and a short description.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The project report(s) as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Script to turn raw data into features for modeling
    │   │   └── 3.0-Preprocessing.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── 3.2-Modeling.py
    │       └── 1.1-Stacking_Approach.py
 

--------

## Project Overview

This repository hosts a dataset containing detailed information about solar power plants in California, with a focus on their geographic distribution, technical characteristics, and proximity to critical energy infrastructure such as substations. The dataset is particularly valuable for:

- **Energy Planning**: Understanding the spatial distribution of solar projects to optimize grid integration and expansion.
- **Renewable Energy Development**: Identifying areas with high solar potential for future development.
- **Spatial Analysis**: Leveraging geographic information system (GIS) data for advanced spatial modeling and visualization.

In our approach we focus on creating a binary classification machine learning model which, based on this dataset, predicts the feasibility of solar power plant installations for solar energy.

The dataset comprises **5,397 rows** and **21 columns** (excluding the `OBJECTID` index column).

---

## Dataset Description

### Columns

The dataset includes the following columns, each providing specific insights into the characteristics of solar power plants in California:

| **Column Name**                              | **Description**                                                                                  | **Data Type** | **Notes**                              |
|----------------------------------------------|--------------------------------------------------------------------------------------------------|---------------|----------------------------------------|
| `County`                                     | Name of the Californian county where the data point is located                                   | String        |                                        |
| `Acres`                                      | Area in acres (1 ac = 4046.8564224 m²) associated with each data point                           | Float         |                                        |
| `Install Type`                               | Type of solar installation (modes: 'Rooftop', 'Parking', 'Ground')                               | String        |                                        |
| `Urban or Rural`                             | Classification of the location as urban or rural (modes: 'Urban', 'Rural')                       | String        |                                        |
| `Combined Class`                             | Combination of `Install Type` and `Urban or Rural` (e.g., 'Rooftop - Urban', 'Ground - Rural')   | String        |                                        |
| `Distance to Substation (Miles) GTET 100 Max Voltage` | Distance to a substation with GTET 100 Max Voltage, in miles (1 mi = 1,609.344 m)               | Float         |                                        |
| `Percentile (GTET 100 Max Voltage)`          | Percentile ranking related to GTET 100 Max Voltage distance                                      | String        |                                        |
| `Substation Name GTET 100 Max Voltage`       | Name of the GTET 100 Max Voltage substation                                                      | String        |                                        |
| `HIFLD ID (GTET 100 Max Voltage)`            | Unique identifier for the GTET 100 Max Voltage substation                                        | Float         | Contains missing values                |
| `Distance to Substation (Miles) GTET 200 Max Voltage` | Distance to a substation with GTET 200 Max Voltage, in miles (1 mi = 1,609.344 m)               | Float         |                                        |
| `Percentile (GTET 200 Max Voltage)`          | Percentile ranking related to GTET 200 Max Voltage distance                                      | String        |                                        |
| `Substation Name GTET 200 Max Voltage`       | Name of the GTET 200 Max Voltage substation                                                      | String        |                                        |
| `HIFLD ID (GTET 200 Max Voltage)`            | Unique identifier for the GTET 200 Max Voltage substation                                        | Float         | Contains missing values                |
| `Distance to Substation (Miles) CAISO`       | Distance to a CAISO substation, in miles (1 mi = 1,609.344 m)                                    | Float         |                                        |
| `Percentile (CAISO)`                         | Percentile ranking related to CAISO substation distance                                          | String        |                                        |
| `Substation CASIO Name`                      | Name of the CAISO substation                                                                     | String        | Contains a few missing values           |
| `HIFLD ID (CAISO)`                           | Unique identifier for the CAISO substation                                                       | Float         | Contains missing values                |
| `Solar Technoeconomic Intersection`          | Indicates technoeconomic intersection for solar energy (modes: 'Within', 'Outside')              | String        | Refers to areas with high/low solar potential |
| `Shape__Area`                                | Area of the geographical shape, in square meters                                                 | Float         | Useful for GIS applications            |
| `Shape__Length`                              | Length of the geographical shape                                                                 | Float         | Useful for GIS applications            |

### Key Features

- **Location Information**: Includes county names and urban/rural classifications, enabling geographic segmentation and analysis.
- **Proximity to Substations**: Provides distances to high-voltage substations (≥100 kV and ≥200 kV) and CAISO substations, along with percentile rankings, facilitating grid integration studies.
- **Solar Installation Details**: Captures the type of solar installation (e.g., rooftop, parking, ground) and associated land area in acres.
- **Spatial Data**: Includes `Shape__Area` and `Shape__Length` for advanced GIS applications and spatial modeling.
- **Technoeconomic Insights**: The `Solar Technoeconomic Intersection` column highlights areas with high or low solar potential, supporting feasibility studies.

### Target Variable

The `Solar Technoeconomic Intersection` column serves as a potential target variable for predictive modeling and analysis. It classifies areas as 'Within' or 'Outside' regions of high solar technoeconomic feasibility, making it ideal for:

- Predicting suitability for solar energy generation.
- Supporting decision-making in energy infrastructure planning.
- Optimizing resource allocation for renewable energy projects.

###  Data Quality Notes
Missing Values: Some columns, particularly HIFLD IDs, contain missing values. Consider appropriate imputation or handling techniques during analysis.
Last Updated: Dataset was last updated in July 2024.

---

## Use Cases

This dataset is versatile and can be applied to a range of research, policy, and industry applications, including but not limited to:

1. **Energy Infrastructure Planning**:
   - Analyze the proximity of solar projects to substations to optimize grid connectivity and expansion.
   - Identify underserved regions for targeted renewable energy development.

2. **Renewable Energy Development**:
   - Assess the technoeconomic feasibility of new solar projects based on geographic and spatial attributes.
   - Prioritize areas with high solar potential for investment and policy incentives.

3. **Spatial Analysis and GIS**:
   - Leverage `Shape__Area` and `Shape__Length` for advanced spatial modeling and visualization.
   - Study the distribution of solar projects across urban and rural areas.

4. **Policy Analysis**:
   - Evaluate the impact of urban/rural classification on solar project deployment.
   - Inform policy decisions related to renewable energy subsidies and zoning regulations.

5. **Machine Learning and Predictive Modeling**:
   - Use the `Solar Technoeconomic Intersection` column as a target variable for classification models.
   - Predict optimal locations for future solar installations based on geographic and technical features.

---

## Installation

To use this dataset, fork the repository and ensure you have the necessary tools for data analysis. Below are the steps to get started:

### Prerequisites

- **Python 3.8+**: For data analysis and visualization.
- **GIS Software (Optional)**: Tools like QGIS or ArcGIS for spatial analysis.
- **Dependencies**: Install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Loading the Dataset

```python
import pandas as pd

# Load the dataset
path1 = r"..\data\raw\Solar_Footprints_V2_7811899327930675815.csv"
path2 = r"..\data\raw\Population-Density By County.csv"

dfs = pd.read_csv(path1, index_col="OBJECTID") # Data frame for solar data
dfp = pd.read_csv(path2) # Data frame for US population data

# Filter rows with "California"
dfp = dfp[dfp["GEO.display-label"].str.contains("California", na=False)]

# Display basic information
print("Solar DataFrame Shape:", dfs.shape)
display("Solar DataFrame:", dfs.head())
print("Population DataFrame Shape:", dfp.shape)
display("Population DataFrame:", dfp.head())
```
## Acknowledgments

We would like to express our gratitude to the following organizations and individuals who made this project possible:

- **California Energy Commission (CEC)**: For providing access to open-source data on solar power plants and energy infrastructure.
- **California Natural Resources Agency**: For giving access to the corresponding interactive geographic map (https://gis.data.cnra.ca.gov/datasets/CAEnergy::solar-footprints-in-california/about).
- **Open Source Community**: For the tools and libraries (e.g., Geopandas, Plotly, Scikit-learn, SHAP, Optuna, ...) that enabled data processing, analysis, and visualization.

If you use this dataset in your work, please consider citing this repository or acknowledging the original data sources.

---

## Contact

For questions, feedback, or collaboration opportunities, please reach out to the project maintainers:

- **GitHub**:
   - [ManuelMonterrosas](https://github.com/ManuelMonterrosas)
   - [tortoise23](https://github.com/tortoise23)
   - [Impuls101](https://github.com/Impuls101)

We welcome contributions, suggestions, and ideas to improve this dataset and make it more useful for the community!

---

Thank you for exploring the **Solar Power Plants Dataset - California** repository! We hope this dataset proves valuable for your research, analysis, or energy planning initiatives. 🌞
