# mnist_serving_pytorch
mnist serving example using pytorch 2.0 & flask

### install libraries
```
pip3 install -r requirements.txt
```

### train
Training mnist with simple network and save a model
```
python3 train.py
```

### serving using flask
serving model using flask
```
python3 flask_server.py #if you want to execute at background: nohup python3 flask_server.py &
```

### test
You can test rest api.
```
python3 flask_test.py
```

<br>
https://seokhyun2.tistory.com/43
