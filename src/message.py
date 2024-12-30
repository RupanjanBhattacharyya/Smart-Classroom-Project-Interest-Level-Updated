class Message:
    '''Sends model data to firestore database'''
    def __init__(self):
        self.poses = []
        self.avg_attention = 0
        self.total_time_eyes_closed = 0
        self.total_time_mouth_open = 0
        self.count = 0
        self.yawn_duration = 0
        self.pose = None
        self.cur_time = 0

    def update(self, pose, attn, time_eyes_closed, time_mouth_open):
        if pose is not None:
            self.poses.append(pose)
        self.avg_attention += attn
        self.total_time_eyes_closed = max(time_eyes_closed, self.total_time_eyes_closed)
        self.total_time_mouth_open = max(time_mouth_open, self.total_time_mouth_open)
        self.count += 1

    def reset(self):
        self.poses = []
        self.avg_attention = 0
        self.total_time_eyes_closed = 0
        self.total_time_mouth_open = 0
        self.count = 0
        self.yawn_duration = 0

    def finalize(self, cur_time):
        self.avg_attention /= self.count
        #self.total_time_eyes_closed /= self.count
        #self.total_time_mouth_open /= self.count
        self.pose = None
        if self.poses:
            self.pose = self.poses[len(self.poses)//2]
        self.cur_time = cur_time
        return self

    def as_dict(self):
        return {'time' : self.cur_time, 'attention' : self.avg_attention, 'pose' : self.pose, 'eyes_closed' : self.total_time_eyes_closed, 'mouth_open' : self.total_time_mouth_open}