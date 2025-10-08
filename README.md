# USGS Earthquake Intelligence System

## 📋 Project Overview

The **USGS Earthquake Intelligence System** is a comprehensive real-time data analytics and machine learning platform designed to collect, process, analyze, and predict earthquake patterns globally. The system integrates data from the USGS Earthquake API, processes it using distributed computing with Apache Spark, and applies advanced NLP and machine learning techniques to enable semantic search, earthquake prediction, clustering analysis, and time series forecasting. It features a RESTful API built with FastAPI that provides 15+ endpoints for data retrieval, predictions, and alerts, along with interactive visualizations using Plotly for geospatial mapping and statistical analysis.

The platform implements a complete end-to-end pipeline including ETL processing with PySpark, natural language processing using Spark NLP for text analysis, FAISS-powered vector database for semantic search across earthquake descriptions, and multiple machine learning models (Gradient Boosting for magnitude prediction, Random Forest for severity classification, DBSCAN/K-Means for hotspot clustering, and ARIMA for frequency forecasting). The system generates comprehensive reports, manages an alert system for significant seismic events (magnitude 6.0+), and provides interactive data exploration tools through command-line menus. Designed for disaster management, seismological research, and public safety applications, it can be deployed on Google Colab, local environments, or cloud platforms (AWS, GCP, Azure) with Docker support.

---

## 🛠️ Technologies & Tools

### **Big Data & Distributed Computing**
- **Apache Spark 3.5.0** - Distributed data processing engine
- **PySpark** - Python API for Spark (DataFrames, SQL, transformations)

### **Natural Language Processing**
- **Spark NLP 5.1.4** - Text processing pipeline (DocumentAssembler, Tokenizer, SentenceDetector)

### **Vector Database & Semantic Search**
- **FAISS 1.7.4** - Facebook AI Similarity Search (IndexFlatL2)
- **Sentence Transformers 2.2.2** - Text embeddings (all-MiniLM-L6-v2 model)

### **Machine Learning & Statistical Analysis**
- **Scikit-learn 1.3.0** - ML algorithms:
  - RandomForestClassifier (severity classification)
  - GradientBoostingRegressor (magnitude prediction)
  - DBSCAN (hotspot clustering)
  - KMeans (regional clustering)
  - StandardScaler, LabelEncoder
- **Statsmodels 0.14.0** - Time series analysis (ARIMA, seasonal decomposition)
- **Joblib 1.3.2** - Model serialization

### **Data Processing & Analysis**
- **Pandas 2.0.3** - Data manipulation and analysis
- **NumPy 1.24.3** - Numerical computing and array operations

### **Web API Development**
- **FastAPI 0.104.1** - Modern async REST API framework
- **Uvicorn 0.24.0** - ASGI server
- **Pydantic 2.5.0** - Data validation and serialization
- **Pyngrok 7.0.1** - Public URL tunneling for Colab
- **nest-asyncio 1.5.8** - Nested event loop support

### **Data Visualization**
- **Plotly 5.17.0** - Interactive visualizations:
  - scatter_geo (global earthquake maps)
  - scatter, histogram, bar, box plots
- **Matplotlib 3.7.1** - Static plotting
- **Seaborn 0.12.2** - Statistical visualization

### **Data Sources & I/O**
- **Requests 2.31.0** - HTTP library for USGS API calls
- **USGS Earthquake API** - Real-time seismic data source
- **openpyxl 3.1.2** - Excel file handling

### **Python Standard Libraries**
- **json** - JSON data handling
- **datetime/timedelta** - Date and time operations
- **typing** - Type hints (List, Dict, Optional)
- **warnings** - Warning control
- **os, glob, shutil** - File system operations

### **Development Environment**
- **Google Colab** - Cloud-based Jupyter notebook environment
- **Python 3.10+** - Programming language

---

## 📦 Key Features

✅ **Real-time Data Collection** - USGS API integration with 30+ earthquake attributes  
✅ **ETL Pipeline** - PySpark distributed processing with schema validation  
✅ **NLP Processing** - Spark NLP text analysis and entity extraction  
✅ **Semantic Search** - FAISS vector database with natural language queries  
✅ **ML Predictions** - Magnitude prediction and severity classification  
✅ **Clustering Analysis** - DBSCAN hotspot detection and K-Means regional grouping  
✅ **Time Series Forecasting** - ARIMA-based 7-day earthquake frequency prediction  
✅ **Alert System** - Automated alerts for significant earthquakes (magnitude 6.0+)  
✅ **REST API** - 15+ FastAPI endpoints for data access and predictions  
✅ **Interactive Visualizations** - Plotly maps and statistical dashboards  
✅ **Comprehensive Reports** - Automated report generation in multiple formats  
✅ **Export Functionality** - CSV, JSON, Excel export capabilities

---

## 🏗️ System Architecture

```
Data Collection (USGS API)
          ↓
ETL Processing (PySpark)
          ↓
NLP Processing (Spark NLP)
          ↓
Vector Database (FAISS + Sentence Transformers)
          ↓
ML Models (Scikit-learn + Statsmodels)
          ↓
REST API (FastAPI)
          ↓
Visualizations (Plotly/Matplotlib) + Reports
```

---

## 📊 Performance Metrics

- **Data Processing**: 500 records in < 5 seconds
- **Embedding Generation**: 500 texts in ~7 seconds  
- **Search Latency**: < 100ms for top-5 results
- **API Response**: < 200ms average
- **ML Model Accuracy**: 85-95% (classification), RMSE: 0.3-0.5 (regression)

---

## 🚀 Deployment

**Supported Platforms**: Google Colab, Local Development, AWS, GCP, Azure, Docker

**Environment**: Python 3.10+ with virtual environment or containerization
