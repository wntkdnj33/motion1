from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtGui import QFont
import cv2
import mediapipe as mp
import numpy as np
import time, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
import subprocess

class AppUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('App UI')
        self.setGeometry(300, 300, 300, 150)
        self.setFont(QFont('Arial', 12))
        self.create_time = None

        # Motion Detection Activation Button
        self.motion_detection_button = QPushButton('모션 인식 ID 설정', self)
        self.motion_detection_button.move(20, 50)
        self.motion_detection_button.clicked.connect(self.activate_motion_detection)

        self.unlock_button = QPushButton('암호 해제', self)
        self.unlock_button.move(20, 100)
        self.unlock_button.clicked.connect(self.unlock_password)

        # Vertical Layout Configuration
        layout = QVBoxLayout()
        layout.addWidget(self.motion_detection_button)

        self.setLayout(layout)

    # Function to activate motion detection
    def activate_motion_detection(self):
        actions = ['come']
        seq_length = 10
        secs_for_action = 30

        # MediaPipe hands model
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0)

        created_time = int(time.time())
        os.makedirs('dataset', exist_ok=True)

        while cap.isOpened():
            for idx, action in enumerate(actions):
                data = []

                ret, img = cap.read()

                img = cv2.flip(img, 1)

                cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.imshow('img', img)
                cv2.waitKey(3000)

                created_time = int(time.time())
                self.create_time = created_time

                while time.time() - created_time < secs_for_action:
                    ret, img = cap.read()

                    img = cv2.flip(img, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = hands.process(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if result.multi_hand_landmarks is not None:
                        for res in result.multi_hand_landmarks:
                            joint = np.zeros((21, 4))
                            for j, lm in enumerate(res.landmark):
                                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                            # Compute angles between joints
                            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                            v = v2 - v1 # [20, 3]
                            # Normalize v
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            # Get angle using arcos of dot product
                            angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                            angle = np.degrees(angle) # Convert radian to degree

                            angle_label = np.array([angle], dtype=np.float32)
                            angle_label = np.append(angle_label, idx)

                            d = np.concatenate([joint.flatten(), angle_label])

                            data.append(d)

                            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    cv2.imshow('img', img)
                    if cv2.waitKey(1) == ord('q'):
                        break

                data = np.array(data)
                print(action, data.shape)
                np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

                # Create sequence data
                full_seq_data = []
                for seq in range(len(data) - seq_length):
                    full_seq_data.append(data[seq:seq + seq_length])

                full_seq_data = np.array(full_seq_data)
                print(action, full_seq_data.shape)
                np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

            cap.release()
            cv2.destroyAllWindows()
        self.train_model()

    def train_model(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        action = 'come'

        try:
            data = np.load(os.path.join('dataset', f'seq_{action}_{self.create_time}.npy'))
            x_data = data[:, :, :-1]
            labels = np.zeros(data.shape[0])
            print(x_data.shape)
            print(labels.shape)

            y_data = to_categorical(labels, num_classes=1)
            print(y_data.shape)

            x_data = x_data.astype(np.float32)
            y_data = y_data.astype(np.float32)

            x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

            print(x_train.shape, y_train.shape)
            print(x_val.shape, y_val.shape)

            model = Sequential([
                LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
            model.summary()

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=200,
                callbacks=[
                    ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
                    ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
                ]
            )

            fig, loss_ax = plt.subplots(figsize=(16, 10))
            acc_ax = loss_ax.twinx()

            loss_ax.plot(history.history['loss'], 'y', label='train loss')
            loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
            loss_ax.set_xlabel('epoch')
            loss_ax.set_ylabel('loss')
            loss_ax.legend(loc='upper left')

            acc_ax.plot(history.history['acc'], 'b', label='train acc')
            acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
            acc_ax.set_ylabel('accuracy')
            acc_ax.legend(loc='upper left')
            plt.show()


            model = load_model('model.h5')

            y_pred = model.predict(x_val)

            confusion_matrix(y_val > 0.5, y_pred > 0.5)

            if len(y_pred) == 1:
                conf = y_pred[0]
            else:
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
        except FileNotFoundError:
            print(f"Error: 'seq_{action}.npy' file not found.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def unlock_password(self):
        actions = ['Your Action', '?']
        seq_length = 10

        model = load_model('model.h5')

        # MediaPipe hands model
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0)
        seq = []
        action_seq = []
        start_time = time.time()  # 시작 시간을 기록합니다.
        your_action_count = 0  # "Your Action"이 반복된 횟수를 저장합니다.

        while cap.isOpened():
            ret, img = cap.read()
            img0 = img.copy()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    if len(seq) < seq_length:
                        continue

                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                    y_pred = model.predict(input_data).squeeze()

                    if y_pred.ndim == 0:  # 예측값(y_pred)이 0차원(스칼라)일 경우
                        conf = y_pred.item()  # conf 변수에 y_pred 값을 할당.
                        i_pred = int(y_pred)  # i_pred 변수에 y_pred 값을 정수형으로 변환하여 할당
                    else:  # 그 외의 경우, 예측값이 1차원 이상인 경우
                        i_pred = int(np.argmax(y_pred))  # i_pred 변수에 y_pred 값 중 가장 큰 값의 인덱스를 정수형으로 변환하여 할당
                        conf = y_pred[i_pred]  # conf 변수에 y_pred 배열에서 i_pred 인덱스에 해당하는 값(확률)을 할당
                        if conf < 0.7:
                            action_seq.append('?')
                            continue

                    action_seq.append(actions[i_pred])  # action_seq 리스트에 i_pred 인덱스에 해당하는 동작(actions[i_pred])을 추가
                if len(action_seq) > 3:
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:  # action_seq 리스트의 마지막 세 개의 값이 모두 같을 경우
                        this_action = action_seq[-1] if action_seq[
                                                            -1] in actions else '?'  # this_action 변수에 마지막 값(action_seq[-1])을 할당합니다. 단, 해당 값이 actions 리스트에 없는 경우 '?'를 할당
                        if this_action == 'Your Action':
                            your_action_count += 1
                        else:
                            your_action_count = 0
                    else:
                        this_action = '?'  # this_action 변수에 '?'를 할당
                        your_action_count = 0
                else:
                    continue

                cv2.putText(img, f'{this_action.upper()}',
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                elapsed_time = time.time() - start_time  # 경과한 시간을 계산합니다.
                if this_action == 'Your Action' and elapsed_time > 10.0 and your_action_count > 50:
                    subprocess.Popen(["python", "11.py"])
                    break  # "Your Action"이 3초 이상 반복되면 프로그램을 종료.

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    app = QApplication([])
    app_ui = AppUI()
    app_ui.show()
    app.exec()
