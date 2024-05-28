



class NoHandDetected(Exception):

    def __init__(self):
        super(NoHandDetected, self).__init__("No hand is detected")