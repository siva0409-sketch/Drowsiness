import cv2

print('OpenCV:', cv2.__version__)

backends = [None, getattr(cv2, 'CAP_DSHOW', None), getattr(cv2, 'CAP_MSMF', None), getattr(cv2, 'CAP_ANY', None)]


def probe(id_list=[0,1,2,3]):
    for cid in id_list:
        for backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(cid)
                else:
                    cap = cv2.VideoCapture(cid, backend)
                ok = cap.isOpened()
                print(f'cam={cid} backend={backend} opened={ok}')
                cap.release()
                if ok:
                    return True
            except Exception as e:
                print('error', cid, backend, e)
    return False

ok = probe()
print('Any camera opened:', ok)
if not ok:
    print('Hint: close other camera apps, check Windows camera privacy settings, or run with admin')
