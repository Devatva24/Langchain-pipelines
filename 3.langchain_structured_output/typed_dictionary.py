from typing import TypedDict

class Person(TypedDict):
    name : str
    age : int
newPerson : Person = {'name' : 'JohnDoe', 'age' : '30'} # no validation at runtime

print(newPerson)