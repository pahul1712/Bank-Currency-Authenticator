FROM continuumio/anaconda3:4.4.0

# Set the working directory
WORKDIR /usr/app/

# Copy the application files
COPY . /usr/app/

# Upgrade pip to avoid compatibility issues
RUN python -m pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Expose the application port
EXPOSE 5000

# Run the Flask application
CMD ["python", "flask_api.py"]
