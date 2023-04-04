FROM jupyter/minimal-notebook
WORKDIR /code
COPY req.txt /code/
RUN pip install -r req.txt
COPY . /code/
ENTRYPOINT ["start.sh", "jupyter", "lab", "--LabApp.token=''"]