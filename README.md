# 얼굴 표정 - detector

fer2013 에 기반한 알고리즘으로 webcam 으로 얼굴 영상 입력을 받아 open cv 처리를 거쳐 기 학습된 기계학습 프로그램에 입력, 질의하도록 조금 변형시킨 코드로 ubuntu 16.04, jupyter notebook (python 2.7), tensorflow, open cv2.4.11 에서 작동함.   

face detector.ipynb, fer2013_2.py, fer2013_eval_bc2.py, checkpoint, model.ckpt-8000, haarcascade_frontalface_alt.xml 파일들은 같은 폴더에 있어야함.

model.ckpt-8000 는 학습 코드 fer2013_train.py 에 훈련 데이터 셑 fe2013_data 를 8000 번 이터레이션 시킨 후의 내부 파라미터들 테이터로 checkpoint 파일에 매 1000 번의 경로가 기록되있음.

원본들과 각 데이터셑들은 다음 저장소에서 확인할 수 있음. 
https://drive.google.com/drive/folders/0B3AAd5V37KvzeV96ZDd3dTFJNDA

참고,

https://github.com/isseu/emotion-recognition-neural-networks

https://drive.google.com/drive/folders/0B3AAd5V37KvzeV96ZDd3dTFJNDA
