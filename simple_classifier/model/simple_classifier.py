from simple_classifier.config.cfg import cfg


class Classifier():
    def __call__(self, height):
        """returns confidence of belonging to the class of basketball players"""
        return height / cfg.max_height
