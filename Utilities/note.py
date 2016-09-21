# note.py
# definition of a note class containing information used to train a neural network
# number, start time, stop time and voice are the only attributes that are important to worry about

class Note:
    def __init__(this,number,start_time, stop_time = -1, voice=-1, num_voices = 4):
        this.num = number
        this.start_time = start_time
        this.voice = voice # voices are numbered from bottom up
        this.upper_bound = num_voices-1
        this.lower_bound = 0
        this.stop_time = stop_time
    def __str__(self):
        return str((self.num, self.start_time, self.stop_time, self.voice))