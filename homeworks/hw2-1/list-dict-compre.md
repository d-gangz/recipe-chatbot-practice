## Python Comprehensions Cheat Sheet

Comprehensions let you build a new collection (list, dict, set, etc.) from an iterable in a single expression. They keep the code short while still following the pattern “loop over things → transform them → collect the results.”

### List Comprehensions

Syntax: `[expression for item in iterable if condition]`

- `expression` describes the value to put into the new list.
- `item` is the loop variable.
- `iterable` is whatever you are looping over.
- The optional `if condition` filters items before they reach the output.

Example without a comprehension:

````python
numbers = [1, 2, 3, 4]
 squares = []
for n in numbers:
    squares.append(n * n)
print(squares)  # [1, 4, 9, 16]
Same logic using a list comprehension:

```python
numbers = [1, 2, 3, 4]
squares = [n * n for n in numbers]
````

print(squares) # [1, 4, 9, 16]
Another example with filtering:

```python
words = ["apple", "bee", "car"]
long_words = [word.upper() for word in words if len(word) >= 4]
# Result: ["APPLE", "CAR"]
```

print(long_words) # ['APPLE', 'CAR']

### Dictionary Comprehensions

Syntax: `{key_expression: value_expression for item in iterable if condition}`

The only difference is that each iteration produces a `key: value` pair instead of a single value.

Example without a comprehension:

````python
prices = {"apple": 1.0, "banana": 0.5, "cherry": 2.0}
price_with_tax = {}
for fruit, price in prices.items():  # .items() yields pairs like ('apple', 1.0)
    price_with_tax[fruit] = round(price * 1.08, 2)
print(price_with_tax)  # {'apple': 1.08, 'banana': 0.54, 'cherry': 2.16}
Same logic with a dictionary comprehension:

```python
prices = {"apple": 1.0, "banana": 0.5, "cherry": 2.0}
price_with_tax = {
    fruit: round(price * 1.08, 2)
    for fruit, price in prices.items()  # Unpack each (key, value) pair from the dictionary
}
print(price_with_tax)  # {'apple': 1.08, 'banana': 0.54, 'cherry': 2.16}
````

Another example mapping each number to its square, skipping odd numbers:

```python
numbers = [1, 2, 3, 4]
even_square_map = {
    n: n * n
    for n in numbers
    if n % 2 == 0
}
print(even_square_map)  # {2: 4, 4: 16}
```

### When to Use Which

- Use a **list comprehension** when you need an ordered collection containing the transformed items (e.g., building rows for a DataFrame).
- Use a **dictionary comprehension** when you need quick lookups by key or want to associate each input with a derived value (e.g., mapping Futures to metadata).

Both forms keep the intent clear: they show how the output collection is built right where it is declared. If the expression starts feeling complicated, fall back to a regular `for` loop for readability.

### Bonus: `enumerate`

`enumerate(iterable)` adds an automatic counter when you loop. It yields `(index, item)` pairs so you can get both the position and the value at once.

Manual counter:

```python
letters = ["a", "b", "c"]
index = 0
for letter in letters:
    print(index, letter)
    index += 1
# Output:
# 0 a
# 1 b
# 2 c
```

Same logic using `enumerate`:

```python
letters = ["a", "b", "c"]
for index, letter in enumerate(letters):
    print(index, letter)
# Output:
# 0 a
# 1 b
# 2 c
```

You can even start counting at a different number by passing a second argument:

```python
for index, letter in enumerate(letters, start=1):
    print(index, letter)
# Output:
# 1 a
# 2 b
# 3 c
```

This tool pairs nicely with comprehensions or any loop where both index and value are needed.
