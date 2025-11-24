# Safety Line Data Science & Engineering Technical Test

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.2.4%2B-green.svg)](https://pandas.pydata.org/)
[![Flask](https://img.shields.io/badge/Flask-API-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## 📋 Overview

This repository contains solutions to the Safety Line (SITA FOR AIRCRAFT) Data Science & Engineering technical assessment. The test evaluates practical skills in data processing, machine learning, API development, and deployment workflows commonly used in aviation data science environments.

## 🎯 Project Objectives

The technical test is designed to assess:

- **Data Processing**: Handling time-series flight data and feature engineering
- **Machine Learning**: Building predictive models for fuel flow optimization
- **Software Engineering**: Python packaging and modular code design
- **API Development**: Creating RESTful interfaces for ML models
- **DevOps**: Containerization and deployment readiness

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.7+**: Primary programming language
- **Pandas 1.2.4+**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computations

### API & Deployment
- **Flask**: RESTful API framework
- **Docker**: Containerization for production deployment
- **setuptools**: Python package distribution

### Development Tools
- **Jupyter Notebook**: Interactive development and analysis
- **pickle**: Data serialization for flight signals

## 📁 Project Structure

```
.
├── question1/
│   ├── signals_altitude.pkl      # Flight altitude data (feet)
│   ├── signals_fuel_flow.pkl     # Fuel consumption data (lb/s)
│   ├── signals_speed.pkl         # Aircraft speed data (km/h)
│   ├── signals_wind.pkl          # Wind speed data (knots)
│   └── analysis_notebook.ipynb   # Data exploration & modeling
│
├── question2/
│   ├── question2.py              # ML module for packaging
│   ├── setup.py                  # Package configuration
│   ├── api.py                    # Flask API implementation
│   ├── Dockerfile                # Container specification
│   └── requirements.txt          # Python dependencies
│
└── README.md
```

## 🚀 Question 1: Fuel Flow Modeling

### Problem Statement
Analyze simulated flight signals to build predictive models for fuel consumption based on aircraft operational parameters.

### Data Characteristics
- **Format**: Pandas DataFrames stored as pickle files
- **Structure**: 
  - Rows (index): Time vectors (variable length per flight)
  - Columns: Individual flight recordings
- **Parameters**:
  - Fuel Flow: pounds per second
  - Altitude: feet
  - Speed: kilometers per hour
  - Wind: knots

### Modeling Objectives

#### Model 1: Speed-Based Fuel Flow
- **Fixed Parameter**: Altitude at 8,000 ft
- **Variable**: Speed (effective range)
- **Analysis**: Relationship between airspeed and fuel consumption at cruise altitude

#### Model 2: Altitude-Based Fuel Flow
- **Fixed Parameter**: Speed at 665 km/h
- **Variable**: Altitude (0-15,000 ft range)
- **Analysis**: Impact of climb/descent phases on fuel efficiency

#### Model 3 (Bonus): Multi-Variable Model
- **Variables**: Both speed and altitude
- **Goal**: Comprehensive fuel flow prediction across operational envelope

### Key Questions Addressed
- Feature selection and engineering for time-series flight data
- Handling variable-length temporal sequences
- Wind speed considerations (and why it may be excluded)
- Model validation for aviation safety applications

## 📦 Question 2: Production Pipeline

### 2.1 Python Package Development
**Objective**: Transform ML module into installable Python package

**Deliverable**: `.tar.gz` archive installable via pip

**Key Components**:
- Modular code structure
- Proper package metadata
- Dependency management
- Installation scripts

### 2.2 Flask API Development
**Objective**: Create RESTful interface for ML models

**API Endpoints**:

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Health check | None |
| `/functions` | GET | List available functions | None |
| `/process` | POST | Generic ML pipeline | `n_samples`, `n_features`, `problem` |
| `/classification/process` | POST | Classification workflow | `n_samples`, `n_features` |
| `/regression/process` | POST | Regression workflow | `n_samples`, `n_features` |

**Response Format**: JSON with model statistics, error metrics, and predictions

### 2.3 Docker Containerization
**Objective**: Package API for production deployment

**Deliverables**:
- Dockerfile with multi-stage build
- docker-compose.yml (optional)
- Production-ready configuration

### 2.4 Scalability Considerations
**Challenge**: Handle high-concurrency API access

**Solutions Evaluated**:
- **Gunicorn**: Multi-worker WSGI server
- **uWSGI**: High-performance application server
- **Celery**: Distributed task queue for async processing
- **FastAPI**: Async-native alternative to Flask
- **Kubernetes**: Container orchestration for horizontal scaling

## 🔧 Installation & Usage

### Prerequisites
```bash
python >= 3.7
pandas >= 1.2.4
```

### Question 1: Run Analysis
```bash
cd question1
jupyter notebook analysis_notebook.ipynb
```

### Question 2: Install Package
```bash
cd question2
pip install dist/question2-package.tar.gz
```

### Run Flask API
```bash
python api.py
# API available at http://localhost:5000
```

### Docker Deployment
```bash
docker build -t sfa-ml-api .
docker run -p 5000:5000 sfa-ml-api
```

## 🎓 Skills Demonstrated

- ✅ Time-series data preprocessing
- ✅ Feature engineering for aviation data
- ✅ Regression modeling and validation
- ✅ Python package architecture
- ✅ RESTful API design patterns
- ✅ Containerization best practices
- ✅ Production ML deployment strategies

## 📊 Technical Insights

### Why Exclude Wind Speed?
Wind speed analysis reveals:
- High variability and noise in measurements
- Complex non-linear interactions with other parameters
- Potential multicollinearity with ground/air speed
- Limited predictive power in controlled models

### Model Selection Rationale
- Linear regression for interpretability in safety-critical systems
- Polynomial features for non-linear relationships
- Cross-validation for robust performance estimation

## 🚦 Future Enhancements

- Real-time data streaming integration
- Advanced ensemble models (XGBoost, Random Forest)
- Model monitoring and drift detection
- A/B testing framework for model comparison
- GraphQL API alternative
- Kubernetes deployment manifests

## 📝 License

This project is part of a technical assessment for Safety Line / SITA FOR AIRCRAFT.

---

**Note**: This implementation focuses on demonstrating technical competency in data science, software engineering, and MLOps practices relevant to aviation analytics.
