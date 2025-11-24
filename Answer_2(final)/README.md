# Make sure you are inside the venv // virtual enviroment
## PACKAGE SETUP
```
# install packages
pip3 install -e . 
# create the ml_model_sita_internship-0.1.0.tar.gz in ./dist
python setup.py sdist
#install ml_model_sita_internship-0.1.0.tar.gz as a package so that we can import 
pip install dist/ml_model_sita_internship-0.1.0.tar.gz
```

## Run Flask API
```
python3 app.py 
```
- run some tests to see how good are the Api and if everything is in place.

```
python3 integration_test.py
```

- test one endpoint to check if we are getting the desired result
```
curl "http://localhost:5050/regression/process?n_samples=50&n_features=3" 
```
- do a series of test to check all the functionalities 
```
python3 test_app.py
```


## Docker build and run the instance 
```
./build_docker.sh
docker run -d -p 8080:5055 --name ml_model_api ml-model-api:dev
```