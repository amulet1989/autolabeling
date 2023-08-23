# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run autolabeling.py when the container launches
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--allow-root"]

