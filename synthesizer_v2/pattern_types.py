#TYPE DEFINITIONS
POS = "POS"
WILD = "WILDCARD"
LITERAL = "LITERAL"
OPTIONAL = "OPTIONAL"
ENTITY = "ENTITY"

class stru:
    def __init__(self, type_, value_1, value_2=None):
        self.type_ = type_
        self.value_1 = value_1
        self.value_2 = value_2
