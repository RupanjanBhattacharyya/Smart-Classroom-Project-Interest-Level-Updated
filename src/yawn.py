import time
def update_yawn(mouth_open, time_mouth_open, mouth_already_open, start_mouth_open):
    # Check how long eyes have been closed
    if mouth_open:
        if mouth_already_open:
            time_mouth_open = time.time() - start_mouth_open
        else:
            start_mouth_open = time.time()
            mouth_already_open = True
    else: # eyes opened
        if mouth_already_open:
            mouth_already_open = False
            time_mouth_open = time.time() - start_mouth_open
        else: # eyes not already closed
            time_mouth_open = 0
    return time_mouth_open, start_mouth_open, mouth_already_open