FROM python:3.8

WORKDIR /autism_classifier

COPY . /autism_classifier
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]