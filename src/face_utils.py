import numpy as np

def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=np.float32)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for ii in range(0, 68):
            coords[ii] = (shape.part(ii).x, shape.part(ii).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def angle_to_direction(angle):
    th = 10
    pitch = angle[0] * (180 / np.pi)
    yaw = angle[1] * (180 / np.pi)

    if np.abs(yaw) < 10 and pitch > -th:
        return 'forward'
    
    if yaw > th:
        return 'left'

    if yaw < -th:
        return 'right' 
    
    return 'backward'

def get_mouth_status(shape):
    up_lib = shape[61:64]
    dn_lib = shape[[67, 66, 65]]

    margin = (dn_lib[:, 1] - up_lib[:, 1]).mean()

    if margin > 30:
        return 'open'

    return 'closed'



