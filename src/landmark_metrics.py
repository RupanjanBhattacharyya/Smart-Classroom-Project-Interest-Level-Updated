'''Detect if eyes or mouth are closed for blink detection'''
import math
import numpy as np

def calculate_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    Equivalent to math.dist() introduced in Python 3.8
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

def eye_aspect_ratio(eye: np.array) -> float:
    '''Calculates the eye width to height ratio to indicate whether an eye is closed or not

    Args
    ----
        eye (np.array of floats): The facial landmark coordinates corresponding to the left or right eye

    Returns
    -------
        EAR (float): The computed eye aspect ratio
    '''

    try:
        # EAR = (|p1-p5|+|p2-p4|) / 2*|p0-p3|
        A = abs(calculate_distance(eye[1], eye[5]))
        B = abs(calculate_distance(eye[2], eye[4]))
        C = abs(calculate_distance(eye[0], eye[3]))
        EAR = (A + B) / (2.0 * C)
    except ZeroDivisionError:
        EAR = None

    return EAR

def mouth_aspect_ratio(mouth: np.array) -> float:
    '''Calculates the mouth width to height ratio to indicate whether the mouth is closed or not

    Args
    ----
        mouth (np.array of floats): The facial landmark coordinates corresponding to the mouth

    Returns
    -------
        MAR (float): The computed mouth aspect ratio
    '''
    
    try:
        # MAR = (|p2-p8|+|p3-p7|+|p4-p6|) / 2*|p1-p5|
        A = abs(calculate_distance(mouth[13], mouth[19]))
        B = abs(calculate_distance(mouth[14], mouth[18]))
        C = abs(calculate_distance(mouth[15], mouth[17]))
        D = abs(calculate_distance(mouth[12], mouth[16]))
        MAR = (A + B + C) / (2.0 * D)
    except ZeroDivisionError:
        MAR = None

    return MAR