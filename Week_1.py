#List
my_list = [1,2,3,"Sayed"]
print(my_list[0])
my_list.append("Hossny") # add element to end of list
print(my_list)

#Tuple
# similar to list but its content can not be changed
my_tuple = (1,2,3,"Sayed")
print(my_tuple[2])
tuple_length = len(my_tuple)

#Set
my_set = {1,2,3,2,1} #the set prevents repeating elements
print(my_set)  #output : {1, 2, 3}

#Dictionary
my_dict ={"name":"sayed","age":22}
print(my_dict["name"])
my_dict["ID"]=12 #add element to end
print(my_dict)
my_dict.keys() #returns all keys names
my_dict.values() #returns all values
#we can create more data structures with dictionaries
students = {
    "student1": {"name": "sayed", "age": 21, "grades": {"arabic": 99, "math": 90, "science": 95}},
    "student2": {"name": "sama", "age": 24, "grades": {"arabic": 80, "math": 84, "science": 77}},
}

print(students["student1"])
print(students["student1"]["name"])
print(students["student1"]["grades"])
print(students["student1"]["grades"]["math"])

#Conditional statements (if.. elif.. else)
a = 4
b = 5

if a > b:
    print("a greater than b")
elif b > a:
    print("a less than b")
else:
    print("a equals b")

#While loop
counter = 0
while counter <= 5:
    print(counter)
    counter+=1
    if counter == 4:
        break #breaks the loop

#For loop
my_list = [1,2,3,'a','b','c']
for i in my_list:
    print(i,end = "") # end="" space between element (123abc)

for i in range(0,5):
    print(i, end=" ") # 0 1 2 3 4

for i in range(0,10,2):
    print(i, end =",") #output : 0,2,4,6,8,

for i in range(10):
    if i==2:
        continue
    print(i,end = " ") #output : 0 1 3 4 5 6 7 8 9


#Functions
def add(num1,num2):
    return num1+num2

print(add(1,2))


def print_names(*names):    #*names -> variable add element as a tuple
    for name in names:
        print(name, end = " ")

print_names("Elsayed_1","Hossny_1")

def print_names(names):    #*names -> variable add element as a List
    for name in names:
        print(name, end = " ")

print_names(["Elsayed_2","Hossny_2"])