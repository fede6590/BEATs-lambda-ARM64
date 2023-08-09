FROM public.ecr.aws/lambda/python:3.8-arm64

# Copy function code and models into /var/task
COPY app.py ${LAMBDA_TASK_ROOT}/
COPY model.pt ${LAMBDA_TASK_ROOT}/

# Install our dependencies
COPY requirements.txt  .
RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

#Copy files
COPY BEATs.py .
COPY backbone.py .
COPY modules.py .
COPY labels.csv .

# Set the CMD to your handler 
CMD [ "app.lambda_handler"]