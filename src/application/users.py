NUM_QUESTIONS = 6 * 2 * 3 * 3

import random

class User:
    def __init__(self, id):
        self.id = str(id)

        self.question_order = list(range(NUM_QUESTIONS))
        random.shuffle(self.question_order)

        self.responses = dict()
        self.interactions = dict()

    def is_authenticated(self):
        return True 

    def is_active(self):
        return True 

    def is_anonymous(self):
        return False 
    
    def get_id(self):
        return self.id
