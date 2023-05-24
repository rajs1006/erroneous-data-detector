# Python image to use.
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN python -m pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8080

# Copy the rest of the working directory contents into the container at /app
COPY . /app

# Start the server when the container launches
CMD ["streamlit", "run", "--server.port", "8080", "src/main/Dashboard.py", "--", "--env", "/app/config/config.env"]
# CMD ["python", "src/main/Main.py", "--env", "/app/config/config.env"]