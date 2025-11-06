class Player:
    def __init__(self, name, average, group_code, family_code, friend_code):
        self.name = name
        self.average = float(average)
        self.group_code = group_code
        self.family_code = family_code
        self.friend_code = friend_code

    def __repr__(self):
        return f"{self.name}({self.average})"
