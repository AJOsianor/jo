

# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "script_name.py"]


flask
pandas
numpy
scikit-learn
matplotlib
seaborn

docker build -t my-app .
docker run -p 5000:5000 my-app


sudo systemctl restart docker

cmd /"C:\Users\USER\Documents\Cleaned_Untitled7.py"

