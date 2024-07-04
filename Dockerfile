FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

# Install the required packages
RUN pip install git+https://github.com/dataloop-ai-apps/dtlpy-lidar.git

COPY /requirements.txt .

RUN pip install -r requirements.txt