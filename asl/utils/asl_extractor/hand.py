import mediapipe as mp
import os.path as osp
import numpy as np
import cv2
import os 

from .exceptions import NoHandDetected

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=10)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

hand_adj = np.loadtxt(osp.join(BASE_PATH, "hand_adj.txt"), delimiter=",")

class HandLandMarksExtractor:
    
    _index: int = 0
    _featrs: list = []

    def __init__(self, index: int = 0):
        self._index = index
        self.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(index = {self._index})"
    
    def from_image(self, img_path: str):

        img = cv2.imread(img_path)

        multi_hand = hands.process(img).multi_hand_landmarks

        if multi_hand:
            for landmark in multi_hand[0].landmark:
                self.append([ landmark.x, landmark.y])
        else:
            raise NoHandDetected()
        

    def append(self, featr: list):
        self._featrs.append(featr)

    @property
    def x(self):
        return np.array(self._featrs)
    
    @x.setter
    def x(self, x: np.ndarray | list):
        if isinstance(x, np.ndarray): x = x.tolist() 
        self._featrs = x 

    def edges_index(self, move: int = 0):
        assert isinstance(move, int)
        rows, cols = np.where(hand_adj != 0)
        stacked = np.vstack((rows, cols)) 
        return stacked + move * 21
    
    def clear(self):
        self._featrs = []


# if __name__ == "__main__":

#     hand = Hand()

#     hand.from_image("./ASL/train/B/B1552.jpg")

#     print(hand.x.shape)