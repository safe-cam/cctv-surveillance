FROM python:3.10-slim-bullseye

LABEL maintainer = "mohitmolela@gmail.com"

WORKDIR /app

COPY . .

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r ./requirements.txt

# COPY ML_models .

# COPY web .

# COPY EDA .

# COPY .github .

# COPY README.md .

# COPY requirements.txt .

# COPY .gitignore .

# COPY .dockerignore .

EXPOSE 8501

# CMD [ "streamlit", "run", "web/streamlit.py", "--server.port", "8501" ]
