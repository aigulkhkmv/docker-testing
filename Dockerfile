# set base image
FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY pyproject.toml .

# install poetry into image
RUN pip install poetry

# install dependencies
RUN poetry install

# copy the content
COPY wine_prediction/ .
COPY models/ .
