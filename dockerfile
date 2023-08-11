FROM public.ecr.aws/lambda/python:3.11-arm64

# Update yum and install libgomp
RUN yum update -y
RUN yum install -y libgomp

# Copy function code and models into /var/task
COPY app.py ${LAMBDA_TASK_ROOT}/

# Update pip and install our dependencies
COPY requirements.txt  .
RUN pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

#Copy files
COPY BEATs.py .
COPY backbone.py .
COPY modules.py .
COPY labels.json .

# Set the CMD to your handler 
CMD [ "app.lambda_handler"]