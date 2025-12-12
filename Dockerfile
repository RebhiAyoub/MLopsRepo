FROM python:3.12

WORKDIR /app


# Copy requirements
COPY requirements_deploy.txt .


RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy only necessary files
COPY app.py .
#COPY loading_data.py .
COPY load_model.py .
#COPY model_pipeline.py .
COPY data_engineering.py .
COPY random_forest_uber_model.joblib .
COPY prepared_data.joblib .
COPY index.html .
#COPY admin.html .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
