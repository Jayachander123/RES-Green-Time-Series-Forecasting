# Dockerfile for the RES Paper Reproducibility Package

# 1. Python base
FROM python:3.11-slim

# 2. Working Directory
WORKDIR /app

# 3. System-level dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements.txt
COPY requirements.txt .

# 5. Install all Python packages.
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple 

# 6. Copy the entire project source code into the container's working directory.
COPY . .

# 7. Make the shell scripts executable.
RUN chmod +x quick_sanity_check.sh run_all_experiments.sh

# 8. Set the default command to run when the container starts.
# CMD ["./quick_sanity_check.sh"]
CMD ["bash"]
