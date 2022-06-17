from re import A
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import asyncio
from torchvision import transforms
import torch
from retina_face import RetinaFace
from gaze_model import gaze_model
from utils import tensor2box
import mediapipe as mp
app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def draw_gaze(image_in, pitchyaw, center, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (center[0], center[1])
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def draw_gaze_boxes(image_in, pitchyaw, center, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    x_c = int(w // 2.0)
    y_c = int(h // 2.0)
    pos = (center[0], center[1])
    
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    
    # dx = np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    # dy = np.sin(pitchyaw[1])
    
    xpad = int(w//4)
    ypad = int(h//4)
    dx = np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = np.sin(pitchyaw[1])
    if dx > 0.2 and dy < -0.2 :
        cv2.rectangle(image_in, (0, ypad*3), (xpad, ypad*4), (0, 0, 255), 2)
    elif dx > 0.2 and -0.2 < dy < 0:
        cv2.rectangle(image_in, (0, ypad*2), (xpad, ypad*3), (0, 0, 255), 2)
    
    elif 0 < dx < 0.2 and dy < -0.2 :
        cv2.rectangle(image_in, (xpad, ypad*3), (xpad*2, ypad*4), (0, 0, 255), 2)
    elif 0 < dx < 0.2 and -0.2 < dy < 0 :
        cv2.rectangle(image_in, (xpad, ypad*2), (xpad*2, ypad*3), (0, 0, 255), 2)
        
    elif dx < -0.2 and dy < -0.2:
        cv2.rectangle(image_in, (xpad*2, ypad*3), (xpad*3, ypad*4), (0, 0, 255), 2)
        
    elif dx < -0.2 and -0.2 < dy < 0:
        cv2.rectangle(image_in, (xpad*2, ypad*2), (xpad*3, ypad*3), (0, 0, 255), 2)
        
    elif -0.2 < dx < 0 and dy < -0.2:
        cv2.rectangle(image_in, (xpad*3, ypad*2), (xpad*4, ypad*3), (0, 0, 255), 2)
    elif -0.2 < dx < 0 and -0.2 < dy < 0:
        cv2.rectangle(image_in, (xpad*3, ypad*2), (xpad*4, ypad*3), (0, 0, 255), 2)
        
    elif dx > 0.2 and dy > 0.2 :
        cv2.rectangle(image_in, (0, 0), (xpad, ypad), (0, 0, 255), 2)
    elif dx > 0.2 and 0.2 > dy > 0 :
        cv2.rectangle(image_in, (0, ypad), (xpad, ypad*2), (0, 0, 255), 2)        
        
    elif 0.2 > dx > 0 and dx < 0.2 and dy > 0.2 :
        cv2.rectangle(image_in, (xpad, 0), (xpad*2, ypad), (0, 0, 255), 2)
    elif 0.2 > dx > 0 and dx < 0.2 and 0.2 > dy > 0 :
        cv2.rectangle(image_in, (xpad, ypad), (xpad*2, ypad*2), (0, 0, 255), 2)
        
    elif dx < -0.2 and dy > 0.2:
        cv2.rectangle(image_in, (x_c, 0), (x_c+xpad, ypad), (0, 0, 255), 2)
    elif dx < -0.2 and 0.2 > dy > 0:
        cv2.rectangle(image_in, (x_c, ypad), (x_c+xpad, ypad*2), (0, 0, 255), 2)
        
    elif 0 > dx > -0.2 and dy > 0.2:
        cv2.rectangle(image_in, (x_c+xpad, 0), (x_c+(2*xpad), ypad), (0, 0, 255), 2)        
    elif 0 > dx > -0.2 and 0.2 > dy > 0:
        cv2.rectangle(image_in, (x_c+xpad, ypad), (x_c+(2*xpad), ypad*2), (0, 0, 255), 2)      
        
    # elif dx > 0 and dy < 0:
    #     cv2.rectangle(image_in, (0, y_c), (pad, y_c+pad), (0, 0, 255), 2)
    # elif dx > 0 and dy < 0:
    #     cv2.rectangle(image_in, (0, y_c), (pad, y_c+pad), (0, 0, 255), 2)
    # elif dx > 0 and dy < 0:
    #     cv2.rectangle(image_in, (0, y_c), (pad, y_c+pad), (0, 0, 255), 2)
        
    # cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
    #                tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
    #                thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out



import math
import numpy

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

def euler_from_matrix(matrix, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def limit_yaw(euler_angles_head):
    # [0]: pos - roll right, neg -   roll left
    # [1]: pos - look down,  neg -   look up
    # [2]: pos - rotate left,  neg - rotate right
    euler_angles_head[2] += np.pi
    if euler_angles_head[2] > np.pi:
        euler_angles_head[2] -= 2 * np.pi

    return euler_angles_head

def get_phi_theta_from_euler(euler_angles):
    return -euler_angles[2], -euler_angles[1]

cfg = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

net = RetinaFace(cfg, phase='test').cuda()
net = load_model(net, './Resnet50_Final.pth', False)
net.eval()

gaze_net = gaze_model().cuda()

# gaze_net = load_model(gaze_net, './model_best.pt', False)
gaze_net = load_model(gaze_net, './model_best.pt', False)
# gaze_net = load_model(gaze_net, './rt_gene_best.pt', False)
# gaze_net = load_model(gaze_net, './Res50_PureGaze_ETH.pt', False)

gaze_net.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

vector = [0,0]

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            # frame, le, re, center = tensor2box(net, frame)
            # face, center = tensor2box(net, frame)
            frame, face, le, re, center = tensor2box(net, frame)
            face.flags.writeable = False
            # Get the result
            img_h, img_w, img_c = frame.shape
            results = face_mesh.process(frame)
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])       
                        
                        # Convert it to the NumPy array
                        face_2d = np.array(face_2d, dtype=np.float64)

                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w

                        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                                [0, focal_length, img_w / 2],
                                                [0, 0, 1]])

                        # The distortion parameters
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        
                        _rotation_matrix, _ = cv2.Rodrigues(rot_vec)
                        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
                        _m = np.zeros((4, 4))
                        _m[:3, :3] = _rotation_matrix
                        _m[3, 3] = 1
                        # Go from camera space to ROS space
                        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                                        [-1.0, 0.0, 0.0, 0.0],
                                        [0.0, -1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]]
                        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
                        # roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

                        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)
        
        
                        # Convert it to the NumPy array
                        # To improve performance
                        face = torch.from_numpy(np.float32(face).transpose(2, 0, 1)).unsqueeze(0).cuda()
                        
                        le = torch.from_numpy(np.float32(le).transpose(2, 0, 1)).unsqueeze(0).cuda()
                        re = torch.from_numpy(np.float32(re).transpose(2, 0, 1)).unsqueeze(0).cuda()
                        head_pose = torch.from_numpy(np.float32([phi_head, theta_head])).unsqueeze(0).cuda()
                            
                        b,c,w,h = face.size()
                        b,c,lw,lh = le.size()
                        b,c,rw,rh = re.size()
                        if w > 1 and h > 1 and lw > 1 and lh > 1 and rw > 1 and rh > 1  :
                            face = torch.nn.functional.interpolate(face, size=(224,224), scale_factor=None,
                                mode='nearest', align_corners=None,)
                            
                            
                            data = {
                                        'face' : face.to('cuda:0'),
                                        'left' : le.to('cuda:0'),
                                        'right' : re.to('cuda:0'),
                                        'head_pose' : head_pose.to('cuda:0')
                                    }
                            pred = gaze_net(data).cpu().detach()[0]
                            
                            global vector
                            vector = pred
                            
                            global pp_vector
                            ppred = pred - pp_vector
                            
                            frame = draw_gaze(frame, [pred[0], pred[1]], center)
                            frame = draw_gaze(frame, [ppred[0], ppred[1]], center, color=(255,0,0))
                            # frame = draw_gaze(frame, [0.1, 0.1], center, color=(0,255,0))
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# @app.route('/inference', methods=['POST'])
# def inference():
#     data = request.json
#     result = face_model.forward(normalize(np.array(data, dtype=np.uint8)))
#     return str(result.item())

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

pp_vector = torch.from_numpy(np.float32([0, 0]))

@app.route('/post', methods=['GET','POST'])
def post():
    return render_template('testing.html')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')



def gen_frames2():  # generate frame by frame from camera
    cap = cv2.VideoCapture('./videoplayback.mp4')
    while(cap.isOpened()):
        ret, t_frame = cap.read()
        success, frame = camera.read()  # read the camera frame
        # frame = cv2.flip(frame, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not success:
            break
        else:
            # frame, le, re, center = tensor2box(net, frame)
            # face, center = tensor2box(net, frame)
            frame, face, le, re, center = tensor2box(net, frame)
            face.flags.writeable = False
            # Get the result
            img_h, img_w, img_c = frame.shape
            results = face_mesh.process(frame)
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])       
                        
                        # Convert it to the NumPy array
                        face_2d = np.array(face_2d, dtype=np.float64)

                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w

                        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                                [0, focal_length, img_w / 2],
                                                [0, 0, 1]])

                        # The distortion parameters
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        
                        _rotation_matrix, _ = cv2.Rodrigues(rot_vec)
                        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
                        _m = np.zeros((4, 4))
                        _m[:3, :3] = _rotation_matrix
                        _m[3, 3] = 1
                        # Go from camera space to ROS space
                        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                                        [-1.0, 0.0, 0.0, 0.0],
                                        [0.0, -1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]]
                        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
                        # roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

                        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)
        
        
                        # Convert it to the NumPy array
                        # To improve performance
                        face = torch.from_numpy(np.float32(face).transpose(2, 0, 1)).unsqueeze(0).cuda()
                        
                        le = torch.from_numpy(np.float32(le).transpose(2, 0, 1)).unsqueeze(0).cuda()
                        re = torch.from_numpy(np.float32(re).transpose(2, 0, 1)).unsqueeze(0).cuda()
                        head_pose = torch.from_numpy(np.float32([phi_head, theta_head])).unsqueeze(0).cuda()
                            
                        b,c,w,h = face.size()
                        b,c,lw,lh = le.size()
                        b,c,rw,rh = re.size()
                        if w > 1 and h > 1 and lw > 1 and lh > 1 and rw > 1 and rh > 1  :
                            face = torch.nn.functional.interpolate(face, size=(224,224), scale_factor=None,
                                mode='nearest', align_corners=None,)
                            
                            
                            data = {
                                        'face' : face.to('cuda:0'),
                                        'left' : le.to('cuda:0'),
                                        'right' : re.to('cuda:0'),
                                        'head_pose' : head_pose.to('cuda:0')
                                    }
                            pred = gaze_net(data).cpu().detach()[0]
                            
                            # frame = draw_gaze(frame, [pred[0], pred[1]], center)
                            frame = draw_gaze_boxes(t_frame, [pred[0], pred[1]], center)
                            # frame = draw_gaze(frame, [ppred[0], ppred[1]], center, color=(255,0,0))
                            # frame = draw_gaze(frame, [0.1, 0.1], center, color=(0,255,0))
                            ret, buffer = cv2.imencode('.jpg', frame)
                            frame = buffer.tobytes()
                        
                            yield (b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':

    app.run(debug=True)
    