FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Instalar las bibliotecas gráficas necesarias
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Instalar Python y pip
RUN apt-get install -y python3.10 python3-pip

# Update pip
RUN pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run autolabeling.py when the container launches
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--allow-root"]



