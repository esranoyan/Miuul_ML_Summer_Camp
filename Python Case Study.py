# Mission 1: Examine the data structures of the given values.

x = 8 
y = 3.2 
z = 8j + 18 
a = "Hello Space" 
b = True 
c = 23 < 22 
l = [1,2,3,4] 
d = {"Name":"Esra", 
     "Age": 20,
     "Adress":"Space"} 
t = ("Machine Learning", "Data Science") 
s = {"Python", "Machine Learning", "Data Science"} 

print(type(x),type(y),type(z),type(a),type(b),type(c),type(l),type(d),type(t),type(s))

# Mission 2: Convert all letters of the given string to uppercase. 
# Put space instead of comma and period, separate word for word.
# Expected output: ['THE', 'GOAL', 'IS', 'TO', 'TURN', 'DATA', 'INTO', 'INFORMATION,', 'AND', 'INFORMATION', 'INTO', 'INSIGTH.']

text = "The goal is to turn data into information, and information into insigth."

for i in text:
    text = text.replace(","," ").replace("."," ")
    text2 = text.upper()
    text3 = text2.split()
    
print(text3)

# Mission 3: Follow the steps below to the given list.
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Step 1: Look at the number of elements of given of the list.
print(len(lst))
# Step 2: Call the elements at index zero and ten.
print(lst[0], lst[10]) 
# Step 3: Create a list ["D", "A", "T", "A"] from the given list.
list2= lst[0:4]
print(list2)
# Step 4: Delete the element at the eighth index.
lst.pop(8)
print(lst)
# Step 5: Add a new element.
lst.append("R")
print(lst)
# Step 6: Add element "N" again to the eighth index.
lst.insert(8, "N")
print(lst)

# Mission 4: Apply the following steps to the given dict structure.
dict = {'Christian':["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Step 1: Access the key values.
k = dict.keys()
print(k)
# Step 2: Access the values.
v = dict.values()
print(v)
# Step 3: Update the value 12 of the Daisy key to 13.
dict["Daisy"][1] = 13
print(dict)
# Step 4: Add a new value whose key value is Ahmet value [Turkey,24].
dict["Ahmet"] = ["Turkey",24]
print(dict)
# Step 5: Remove Antonio from the dictionary.
dict.pop("Antonio")
print(dict)

# Mission 5: Write a function that takes a list as an argument, assigning the odd and even numbers in the list to separate lists, 
# and returns these lists.

l = [2, 13, 18, 93, 22]
def numbers(num):
    even_list = []
    odd_list = []
    for i in l:
        if i % 2 ==0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

even_list , odd_list = numbers(l)
print(even_list, odd_list)
    
# Mission 6: In the list given below, the names of the students who have entered the degree at the faculty of 
# engineering and medicine is available. Respectively, the first three students represent the order of success 
# of the faculty of engineering, while the last three students also represent the faculty of medicine belongs 
# to the student order. Use Enumerate to print student degrees in a faculty-specific format.

students = ["Ali", "Veli", "Ayse", "Talat", "Zeynep", "Ece"]

# Expected output:
""" Muhendislik Fakultesi 0. ogrenci: Ali
Muhendislik Fakultesi 1. ogrenci: Veli
Muhendislik Fakultesi 2. ogrenci: Ayse
Tip Fakultesi 3. ogrenci: Talat
Tip Fakultesi 4. ogrenci: Zeynep
Tip Fakultesi 5. ogrenci: Ece """

for index, name in enumerate(students):
    if name in students[0:3]:
        print(f"Muhendislik Fakultesi {index}. ogrenci: {name}")
    else:
        print(f"Tip Fakultesi {index}. ogrenci: {name}")
        
# Mission 7: Three lists given below. In lists there are class code, credit and quota in order. 
# Print course information using zip.

class_code = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
credit = [3, 4, 2, 4]
quota = [30, 75, 150, 25]

# Expected output:
"""Kredisi 3 olan CMP1005 kodlu dersin kontenjani 30 kisidir
Kredisi 4 olan PSY1001 kodlu dersin kontenjani 75 kisidir
Kredisi 2 olan HUK1005 kodlu dersin kontenjani 150 kisidir
Kredisi 4 olan SEN2204 kodlu dersin kontenjani 25 kisidir"""

list = zip(credit, class_code, quota)
for course in list:
    print(f"Kredisi {course[0]} olan {course[1]} kodlu dersin kontenjani {course[2]} kisidir.")
    
# Mission 8: Two sets given below. If the 1st cluster includes the 2nd cluster, specify their common elements. 
# If not, specify the difference of the 2nd set from the 1st set and you are expected to define the function that will print the results.

cluster1 = set(["data", "python"])
cluster2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

# Expected output:
# {'function', 'qcut', 'miuul', 'lambda'}

def check_sets():
    if cluster1.issuperset(cluster2):
        print(cluster1.intersection(cluster2))
    else:
        print(cluster2.difference(cluster1))

check_sets()