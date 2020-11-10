# set base image
FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY pyproject.toml .

# install poetry into image
RUN pip3 install poetry==1.0.0
RUN poetry -V

# install dependencies
RUN ls
RUN poetry install

# copy the content
COPY wine_prediction/models .

ENTRYPOINT poetry run python3 /code/trainig.py
