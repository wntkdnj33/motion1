import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['Your Action', '?']
seq_length = 30

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

start_time = None # 학습한 액션을 시작한 시간
end_time = None # 학습한 액션을 끝낸 시간

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

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            if y_pred.ndim == 0: #예측값(y_pred)이 0차원(스칼라)일 경우
                conf = y_pred.item() #conf 변수에 y_pred 값을 할당.
                i_pred = int(y_pred) #i_pred 변수에 y_pred 값을 정수형으로 변환하여 할당
            else: #그 외의 경우, 예측값이 1차원 이상인 경우
                i_pred = int(np.argmax(y_pred))  #i_pred 변수에 y_pred 값 중 가장 큰 값의 인덱스를 정수형으로 변환하여 할당
                conf = y_pred[i_pred] #conf 변수에 y_pred 배열에서 i_pred 인덱스에 해당하는 값(확률)을 할당
                if conf < 0.7:
                    action_seq.append('?')
                    continue

            action_seq.append(actions[i_pred]) #action_seq 리스트에 i_pred 인덱스에 해당하는 동작(actions[i_pred])을 추가
        if len(action_seq) > 3:
            if action_seq[-1] == action_seq[-2] == action_seq[-3]: #action_seq 리스트의 마지막 세 개의 값이 모두 같을 경우
                this_action = action_seq[-1] if action_seq[-1] in actions else '?' # this_action 변수에 마지막 값(action_seq[-1])을 할당합니다. 단, 해당 값이 actions 리스트에 없는 경우 '?'를 할당
            else:
                this_action = '?'#  this_action 변수에 '?'를 할당
        else:
            continue

        cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        if len(action_seq) > 3 and all(action_seq[-1] == action_seq[-i] for i in range(1, 4)) and len(seq) > seq_length * 3: # 마지막 3개 프레임의 예측값이 모두 같고, 현재 시퀀스(seq)의 길이가 seq_length*3 이상일 때
            if all([a == this_action for a in action_seq[-12:]]): # 마지막 12개 프레임의 예측값이 모두 현재 동작(this_action)과 같을 경우
                print(f"Action {this_action} completed!") # 'Action [this_action] completed!' 출력
                break

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

#특성,