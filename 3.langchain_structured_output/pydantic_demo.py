from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'john doe' # here we can put default value also when no value is passed
    age: Optional[int] = None
    email: EmailStr # built in email validator
    cgpa: float = Field(gt = 0, lt = 10, default = 6, description='A decimal value of cgpa scored by a student') # here we can also put our custom validations and constraints; we can also put regex validator for phone numbers etc.

new_student = {'age': '32', 'email' : 'lombok@gmail.com'} # here pydanctic is smart enough to type cast the string value to the integer -> type casting is called typed coercion in python

student = Student(**new_student)

print(student)

student_dict = dict(student)

print(student_dict)

student_json = student.model_dump_json()

print(student_json)