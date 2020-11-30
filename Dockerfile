# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /src

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt
# RUN pip install scikit-learn


# copy the content of the local src directory to the working directory
COPY Yelp_01.py .
COPY Yelp_02.py .
COPY Yelp_03.py .
COPY Yelp_all.py .
COPY data_clean data_clean/
# COPY local_script/ .
# COPY test.py .

# command to run on container start
# CMD [ "python", "import os", "os.listdir()" ]

# command to run on container start
CMD [ "python", "./Yelp_all.py" ]
# CMD [ "python", "./Yelp_03.py" ]