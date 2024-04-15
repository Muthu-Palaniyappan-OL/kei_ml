import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import Model

model = Model()
model.load_state_dict(torch.load("model.pth"))
model.eval()

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (192, 192))
    frame = frame / 255
    tensor_frame = torch.tensor(frame).permute(2, 0, 1).float()
    grid_size = 11

    with torch.no_grad():
        output = model(tensor_frame.unsqueeze(0))
        output = torch.sigmoid(output)
        output = np.array(output.numpy())[0].reshape(11, 11)

    cv2.imshow("Input", original)
    cv2.imshow("Output", cv2.resize(np.float32(output), (192, 192)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
