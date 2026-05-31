import os
import cv2
import time
from threading import Thread
from queue import Queue, PriorityQueue
from loguru import logger

class VideoReader:
    def __init__(self, path, queue_size=64):
        self.cap = cv2.VideoCapture(path)
        self.Q = Queue(maxsize=queue_size)
        self.stopped = False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        frame_id = 0
        while True:
            if self.stopped: return
            if not self.Q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return
                self.Q.put((frame_id, frame))
                frame_id += 1
            else:
                time.sleep(0.001)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.cap.release()

class VideoWriter:
    def __init__(self, path, fps, size):
        self.size = (int(size[0]), int(size[1]))
        save_path = os.path.splitext(path)[0] + ".mp4"
        self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        self.Q = PriorityQueue()
        self.stopped = False
        self.path = save_path
        self.next_frame_id = 0

    def start(self):
        t = Thread(target=self.write, args=())
        t.daemon = True
        t.start()
        return self

    def write(self):
        while True:
            if self.stopped and self.Q.empty(): break
            if not self.Q.empty():
                frame_id, frame = self.Q.get()
                if frame_id < self.next_frame_id: continue
                if frame_id > self.next_frame_id:
                    self.Q.put((frame_id, frame))
                    time.sleep(0.001)
                    continue
                if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                    frame = cv2.resize(frame, self.size)
                self.writer.write(frame)
                self.next_frame_id += 1
            else:
                time.sleep(0.001)

    def put(self, frame_id, frame):
        self.Q.put((frame_id, frame))

    def stop(self):
        self.stopped = True
        while not self.Q.empty(): time.sleep(0.1)
        time.sleep(0.5)
        self.writer.release()
