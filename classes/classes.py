from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def findMatches(self, frame):
        pass

class Processor(ABC):
    @abstractmethod
    def process(self, frame):
        pass

class Tracker(ABC):
    @abstractmethod
    def update_count(self, matches, processed, frame_number):
        pass