# Gaze estimation_serving_pytorch
Gaze estimation serving model using pytorch & flask

Face detection : retina face -> [pre_train](https://drive.google.com/file/d/1wXb_MjIKMe8cyrNBZNvdHBGhVBqRYnI9/view?usp=sharing)

Gaze estimation : RT_GENE -> [pre_train](https://drive.google.com/file/d/1GTnigyi4ndfFNo_ahwRJ6umMsBgzf4o5/view?usp=sharing)


example video: [down](https://drive.google.com/file/d/1mACHbcW3HdyIE4roVfZjAQVMnwFs0hlt/view?usp=sharing)

### install libraries
```
pip install -r requirements.txt
```

### test
```
python flask_server.py 
```

### test
```
각 가중치를 받아 해당 폴더에 위치한후 
python flack_server.py 실행

- 시선추정 결과 백터로 시각화
http://127.0.0.1:5000/

-강의 동영상 좌표 라벨링 테스트
http://127.0.0.1:5000/post

```
