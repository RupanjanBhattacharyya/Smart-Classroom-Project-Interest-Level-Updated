import time
def update_eyes(eyes_closed, time_eyes_closed, eyes_already_closed, start_eyes_closed):
    # Check how long eyes have been closed
    if eyes_closed:
        if eyes_already_closed:
            time_eyes_closed = time.time() - start_eyes_closed
        else:
            start_eyes_closed = time.time()
            eyes_already_closed = True
    else: # eyes opened
        if eyes_already_closed:
            eyes_already_closed = False
            time_eyes_closed = time.time() - start_eyes_closed
        else: # eyes not already closed
            time_eyes_closed = 0
    return time_eyes_closed, start_eyes_closed, eyes_already_closed