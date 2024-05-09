# install pytorch, ollama and uses it with rocm
FROM rocm/pytorch:latest
# COPY . /tg1
COPY chatbot_ollama /tg1
LABEL authors="isaque"
RUN cd /tg1 &&\
    pip install ollama &&\
    curl -fsSL https://ollama.com/install.sh | sh &&\
    pip install -r requirements.txt
ENTRYPOINT ["top", "-b"]
