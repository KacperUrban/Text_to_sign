FROM jupyter/tensorflow-notebook

WORKDIR /code

RUN pip install jupyterlab-git

COPY . /code

ENV JUPYTER_ENABLE_LAB=yes

ENTRYPOINT ["start.sh", "jupyter", "lab", "--LabApp.token=''"]