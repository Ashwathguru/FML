#add_numbers is a function that takes two numbers and adds them together.

def add_numbers(x, y):
    return x + y

add_numbers(1, 2)


-------------------------------------
#add_numbers updated to take an optional 3rd parameter. 
#Using print allows printing of multiple expressions within a single cell.

def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z

print(add_numbers(1, 2))
print(add_numbers(1, 2, 3))

-------------------------------------
#add_numbers updated to take an optional flag parameter.

def add_numbers(x, y, z=None, flag=False):
    if (flag):
        print('Flag is true!')
    if (z==None):
        return x + y
    else:
        return x + y + z
    
print(add_numbers(1, 2, flag=True))

-------------------------------------
#Assign function add_numbers to variable a.

def add_numbers(x,y):
    return x+y

a = add_numbers
a(1,2)

-------------------------------------

#The Python Programming Language: Types and Sequences
#Use type to return the object's type.
type('This is a string')
str

type(None)
NoneType

type(1)
int

type(1.0)
float

type(add_numbers)

-------------------------------------

#Tuples are an immutable data structure (cannot be altered).

x = (1, 'a', 2, 'b')
type(x)


Lists are a mutable data structure
x = [1, 'a', 2, 'b']
type(x)


Use append to append an object to a list.
x.append(3.3)
print(x)


This is an example of how to loop through each item in the list.
for item in x:
    print(item)


Or using the indexing operator:

i=0
while( i != len(x) ):
    print(x[i])
    i = i + 1

Use + to concatenate lists.
[1,2] + [3,4]


Use * to repeat lists.
[1]*3


Use the in operator to check if something is inside a list.

1 in [1, 2, 3]


Now let's look at strings. Use bracket notation to slice a string.

x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters



This will return the last element of the string.
x[-1]


This will return the slice starting from the 4th element from the end and stopping before the 2nd element from the end.
x[-4:-2]


This is a slice from the beginning of the string and stopping before the 3rd element.
x[:3]


And this is a slice starting from the 4th element of the string and going all the way to the end.

x[3:]


firstname = 'Christopher'
lastname = 'Brooks'

print(firstname + ' ' + lastname)
print(firstname*3)
print('Chris' in firstname)


split returns a list of all the words in a string, or a list split on a specific character.

firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)


Make sure you convert objects to strings before concatenating.
'Chris' + 2


Dictionaries associate keys with values.

x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}

x['Christopher Brooks'] # Retrieve a value by using the indexing operator

{'Bill Gates': 'billg@microsoft.com', 'Christopher Brooks': 'brooksch@umich.edu'}

x['Kevyn Collins-Thompson'] = None
x['Kevyn Collins-Thompson']

{'Bill Gates': 'billg@microsoft.com', 'Kevyn Collins-Thompson': None, 'Christopher Brooks': 'brooksch@umich.edu'}

Iterate over all of the keys:
for name in x:
    print(x[name])

Iterate over all of the values:
for email in x.values():
    print(email)

Iterate over all of the items in the list:
for name, email in x.items():
    print(name)
    print(email)

You can unpack a sequence into different variables:
x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x
fname
lname

Make sure the number of values you are unpacking matches the number of variables being assigned.
x = ('Christopher', 'Brooks', 'brooksch@umich.edu', 'Ann Arbor')
fname, lname, email = x


The Python Programming Language: More on Strings
print('Chris' + 2)
print('Chris' + str(2))

Python has a built in method for convenient string formatting.

sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

The Python Programming Language: Dates and Times

import datetime as dt
import time as tm

time returns the current time in seconds since the Epoch. (January 1st, 1970)

tm.time()
1534591038.82

Convert the timestamp to datetime.
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow
datetime.datetime(2018, 8, 18, 11, 19, 16, 902526)

Handy datetime attributes:
dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime
(2018, 8, 18, 11, 19, 16)

timedelta is a duration expressing the difference between two dates.
delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta

datetime.timedelta(100)

date.today returns the current local date.

today = dt.date.today()
today - delta # the date 100 days ago

datetime.date(2018, 5, 10)

today > today-delta # compare dates

True

The Python Programming Language: Objects and map()

An example of a class in python:
class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location

person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))

Christopher Brooks live in Ann Arbor, MI, USA and works in the department School of Information

Here's an example of mapping the min function between two lists.

store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest

<map at 0x7f7e3005f860>

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']



def split_title_and_name(person):
    
    title = person.split()[0]

    lastname = person.split()[-1]

    return '{} {}'.format(title, lastname)



list(map(split_title_and_name(), people))


Now let's iterate through the map object to see the values.

for item in cheapest:
    print(item)

9.0
11.0
12.34
2.01
---------------------------------------------------------
The Python Programming Language: Lambda and List Comprehensions

Here's an example of lambda that takes in three parameters and adds the first two.

my_function = lambda a, b, c : a + b
my_function(1, 2, 3)
3

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']


def split_title_and_name(person):
    
    return person.split()[0] + ' ' + person.split()[-1]



#option 1


for person in people:
    
    print(split_title_and_name(person) == (lambda x: x.split()[0] + ' ' + x.split()[-1])(person))



#option 2

list(map(split_title_and_name, people)) == list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people))


Let's iterate from 0 to 9 and return the even numbers
my_list = []
for number in range(0, 10):
    if number % 2 == 0:
        my_list.append(number)
my_list

[0, 2, 4, 6, 8]

Now the same thing but with list comprehension

my_list = [number for number in range(0,1000) if number % 2 == 0]
my_list

[0, 2, 4, 6, 8]

def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

times_tables() == [j*i for i in range(10) for j in range(10)]

---------------------------------------------------------

The Python Programming Language: Numerical Python (NumPy)

import numpy as np

Creating Arrays

Create a list and convert it to a numpy array
mylist = [1, 2, 3]
x = np.array(mylist)
x
array([1, 2, 3])

Or just pass in a list directly
y = np.array([4, 5, 6])
y
array([4, 5, 6])

Pass in a list of lists to create a multidimensional array.
m = np.array([[7, 8, 9], [10, 11, 12]])
m
array([[ 7,  8,  9],
       [10, 11, 12]])

Use the shape method to find the dimensions of the array. (rows, columns)
m.shape
(2, 3)

arange returns evenly spaced values within a given interval.
n = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30
n
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

reshape returns an array with the same data with a new shape.
n = n.reshape(3, 5) # reshape array to be 3x5
n

array([[ 0,  2,  4,  6,  8],
       [10, 12, 14, 16, 18],
       [20, 22, 24, 26, 28]])

linspace returns evenly spaced numbers over a specified interval.
o = np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4
o
array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ])

resize changes the shape and size of array in-place.
o.resize(3, 3)
o
array([[ 0. ,  0.5,  1. ],
       [ 1.5,  2. ,  2.5],
       [ 3. ,  3.5,  4. ]])

ones returns a new array of given shape and type, filled with ones.
np.ones((3, 2))
array([[ 1.,  1.],
       [ 1.,  1.],
       [ 1.,  1.]])

zeros returns a new array of given shape and type, filled with zeros.
np.zeros((2, 3))
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])

eye returns a 2-D array with ones on the diagonal and zeros elsewhere.
np.eye(3)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])

diag extracts a diagonal or constructs a diagonal array.
np.diag(y)
np.diag(x)
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3]])

Create an array using repeating list (or see np.tile)
np.array([1, 2, 3] * 3)
array([1, 2, 3, 1, 2, 3, 1, 2, 3])

Repeat elements of an array using repeat.
np.repeat([1, 2, 3], 3)
array([1, 1, 1, 2, 2, 2, 3, 3, 3])

Combining Arrays
p = np.ones([2, 3], int)
p

array([[1, 1, 1],
       [1, 1, 1]])

Use vstack to stack arrays in sequence vertically (row wise).
np.vstack([p, 2*p])
array([[1, 1, 1],
       [1, 1, 1],
       [2, 2, 2],
       [2, 2, 2]])

np.vstack([p, 2*p, 3*p])
array([[1, 1, 1],
       [1, 1, 1],
       [2, 2, 2],
       [2, 2, 2],
       [3, 3, 3],
       [3, 3, 3]])

Use hstack to stack arrays in sequence horizontally (column wise)

np.hstack([p, 2*p])
array([[1, 1, 1, 2, 2, 2],
       [1, 1, 1, 2, 2, 2]])

np.hstack([p, 2*p, 3*p])
array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
       [1, 1, 1, 2, 2, 2, 3, 3, 3]])

Operations

Use +, -, *, / and ** to perform element wise addition, subtraction, multiplication, division and power.
print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]

print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]

Dot Product:

x.dot(y) # dot product  1*4 + 2*5 + 3*6

z = np.array([y, y**2])
z
array([[ 4,  5,  6],
       [16, 25, 36]])
print(len(z)) # number of rows of array
2

Let's look at transposing arrays. Transposing permutes the dimensions of the array.
z = np.array([y, y**2])
z
array([[ 4,  5,  6],
       [16, 25, 36]])

The shape of array z is (2,3) before transposing.

z.shape
(2, 3)

Use .T to get the transpose.
z.T
array([[ 4, 16],
       [ 5, 25],
       [ 6, 36]])

The number of rows has swapped with the number of columns.
z.T.shape

(3, 2)

Use .dtype to see the data type of the elements in the array.
z.dtype
dtype('int64')
Use .astype to cast to a specific type.
z = z.astype('f')
z.dtype
dtype('float32')

Math Functions
Numpy has many built in math functions that can be performed on arrays.

a = np.array([-4, -2, 1, 3, 5])
a.sum()
3
a.max()
5
a.min()
-4
a.mean()
0.60
a.std()
3.26

argmax and argmin return the index of the maximum and minimum values in the array.

a.argmax()
4
a.argmin()
0

Indexing / Slicing
s = np.arange(5)**2
s
array([ 0,  1,  4,  9, 16])

Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.

s[0], s[4], s[-1]
(0, 16, 16)

Use : to indicate a range. array[start:stop]

Leaving start or stop empty will default to the beginning/end of the array.

s[1:5]

s = np.arange(13)**2
s
array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144])


s[0], s[4], s[-1]
(0, 16, 144)
s[1:5]
array([ 1,  4,  9, 16])

Use negatives to count from the back.
s[-4:]
array([ 81, 100, 121, 144])

A second : can be used to indicate step-size. array[start:stop:stepsize]

Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.
s[-5::-2]
array([64, 36, 16,  4,  0])

Let's look at a multidimensional array.
r = np.arange(36)
r.resize((6, 6))
r

array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])

Use bracket notation to slice: array[row, column]
r[2, 2]
14

And use : to select a range of rows or columns
r[3, 3:6]
array([21, 22, 23])

Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.
r[:2, :-1]
array([[ 0,  1,  2,  3,  4],
       [ 6,  7,  8,  9, 10]])

This is a slice of the last row, and only every other element.
r[-1, ::2]
array([30, 32, 34])

We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30. (Also see np.where)
r[r > 30]
array([31, 32, 33, 34, 35])
Here we are assigning all values in the array that are greater than 30 to the value of 30.
r[r > 30] = 30
r
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 30, 30, 30, 30, 30]])

Copying Data
Be careful with copying and modifying arrays in NumPy!
r2 is a slice of r
r2 = r[:3,:3]
r2

array([[ 0,  1,  2],
       [ 6,  7,  8],
       [12, 13, 14]])

Set this slice's values to zero ([:] selects the entire array)
r2[:] = 0
r2
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]])

r has also been changed!
r
array([[ 0,  0,  0,  3,  4,  5],
       [ 0,  0,  0,  9, 10, 11],
       [ 0,  0,  0, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 30, 30, 30, 30, 30]])

To avoid this, use r.copy to create a copy that will not affect the original array
r_copy = r.copy()
r_copy
array([[ 0,  0,  0,  3,  4,  5],
       [ 0,  0,  0,  9, 10, 11],
       [ 0,  0,  0, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 30, 30, 30, 30, 30]])

Now when r_copy is modified, r will not be changed.
r_copy[:] = 10
print(r_copy, '\n')
print(r)
[[10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]] 

[[ 0  0  0  3  4  5]
 [ 0  0  0  9 10 11]
 [ 0  0  0 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]
 [30 30 30 30 30 30]]

Iterating Over Arrays
Let's create a new 4 by 3 array of random numbers 0-9.
test = np.random.randint(0, 10, (4,3))
test
array([[5, 1, 7],
       [0, 1, 9],
       [9, 3, 7],
       [1, 6, 6]])

array([[4, 7, 9],
       [0, 9, 0],
       [6, 7, 0],
       [5, 4, 3]])

Iterate by row:
for row in test:
    print(row)
[4 7 9]
[0 9 0]
[6 7 0]
[5 4 3]


Iterate by index:
for i in range(len(test)):
    print(test[i])
[4 7 9]
[0 9 0]
[6 7 0]
[5 4 3]

Iterate by row and index:
for i, row in enumerate(test):
    print('row', i, 'is', row)
row 0 is [4 7 9]
row 1 is [0 9 0]
row 2 is [6 7 0]
row 3 is [5 4 3]

Use zip to iterate over multiple iterables.
test2 = test**2
test2
array([[16, 49, 81],
       [ 0, 81,  0],
       [36, 49,  0],
       [25, 16,  9]])
for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)

4 7 9] + [16 49 81] = [20 56 90]
[0 9 0] + [ 0 81  0] = [ 0 90  0]
[6 7 0] + [36 49  0] = [42 56  0]
[5 4 3] + [25 16  9] = [30 20 12]



____________________________________

import pandas as pd
pd.Series?

animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

numbers = [1, 2, 3]
pd.Series(numbers)

animals = ['Tiger', 'Bear', None]
pd.Series(animals)

numbers = [1, 2, None]
pd.Series(numbers)

import numpy as np
np.nan == None

np.nan == np.nan

np.isnan(np.nan)

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s

s.index

s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s

------------------------------

Querying a Series:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s

s.iloc[3]

s.loc['Golf']

s[3]

s['Golf']

sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)

s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead

s = pd.Series([100.00, 120.00, 101.00, 3.00])
s

total = 0
for item in s:
    total+=item
print(total)

import numpy as np

total = np.sum(s)
print(total)

s = pd.Series(np.random.randint(0,1000,10000))
s.head()

len(s)

%%timeit -n 100
summary = 0
for item in s:
    summary+=item

%%timeit -n 100
summary = np.sum(s)

s+=2 #adds two to each item in s using broadcasting
s.head()

for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()

%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,10000))
for label, value in s.iteritems():
    s.loc[label]= value+2

%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,10000))
s+=2

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)

original_sports

cricket_loving_countries
all_countries
all_countries.loc['Cricket']

-----------------------------




python-code-for students.txt
Displaying python-code-for students.txt.