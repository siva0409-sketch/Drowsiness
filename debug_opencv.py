import cv2
import sys
import importlib.metadata as m

print('executable', sys.executable)
print('version', cv2.__version__)
print('cv2 file', cv2.__file__)
print('dists', [dist.metadata['Name'] for dist in m.distributions() if 'opencv' in dist.metadata['Name']])
info = cv2.getBuildInformation()
for line in info.splitlines():
    if any(key in line for key in ['Video I/O', 'GUI', 'FFMPEG', 'Media Foundation', 'MSMF', 'DShow']):
        print(line)
