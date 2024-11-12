

# Calling GPT as a decorated python function

## Installation

put `promptium` in your working directory or pythonpath

Create the file `keys/openai` relative to your working directory, with exactly two lines:
1. openai organization id
2. openai api key
(get these in your openai user profile page)

MAKE SURE YOUR KEY DOESNT GET PUSHED TO GITHUB

## Usage

Make a normal python function, decorate it with `@prompt`, and put the prompt template as the docstring. Curly braces `{}` denote prompt template parameters, and by default, the names/positions of the prompt arguments correspond to the names/positions of function parameters and will be filled automatically when the function is called.

```python
from promptium.prompt import prompt

@prompt
def list_pairs_of_stuff(n, generated=None):
    """
    List {n} pairs of everyday objects that go together, formatted like:

    N. <object 1> and <object 2>


    """
    pairs = []
    for line in generated.split("\n"):
        if "." in line:
            a, b = line.split(".")[1].split(' and ')
            pairs.append(a, b)
    return pairs
```

The body of the function defines how to transform the text string generated by GPT (received as the special parameter `generated`) into a return value.

Once you finish defining the prompt function, you can just call it like a function.

```python
for a, b in list_pairs_of_stuff(6):
    print(a, 'goes with', b)
```
Output
```text
Toothbrush goes with toothpaste
Fork goes with knife
Pen goes with paper
Key goes with lock
Scissors goes with tape
Shoes goes with socks
```

### Autosave / Caching

Every time you call GPT in this way, GPT generations will automatically be cached into a json file (named according to the function name) in the folder `llm_cache`. This allows you to avoid ever calling GPT redundantly (since cached generations are retrieved based on the prompt string), and it gives you autosave for everything you generate.

### Custom Prompt Filling

For full control, switch the special `generate` parameter for the special `llm` parameter, and you can call GPT yourself with custom prompt arguments rather than having the prompt template filled automatically.

```python
from promptium.prompt import prompt

@prompt
def list_pairs_of_stuff(n, llm=None):
    """
    List {n} pairs of everyday objects that go together, formatted like:

    N. <object 1> and <object 2>


    """
    generated = llm.generate(max(n, 1))
    pairs = []
    for line in generated.split("\n"):
        if "." in line:
            a, b = line.split(".")[1].split(' and ')
            pairs.append((a, b))
    return pairs

for a, b in list_pairs_of_stuff(6):
    print(a, 'goes with', b)
```

### Logging

Use the `log` parameter with a callable (like `print`) or file path to log generations in human readable format.

```python
list_pairs_of_stuff(6, log=print)
```

### Retries and Backoff

Calling a prompt function automatically retries if the API call fails. Successive failures triggers waiting that maxes out at 1 minute.

### Crashless

By default, calling a prompt function is not allowed to crash. To allow crashing, use the `debug` parameter:

```python
list_pairs_of_stuff(6, log=print, debug=True)
```

### Parallelism

Usually a single process will use about 1,500 tokens / minute, much lower than the API call limit of 90,000 tokens / minute. To go faster, you can use multiprocessing.

```python

from promptium.prompt import prompt
from promptium.parse import parse, list_items
import multiprocessing as mp

@prompt
def list_some(n, thing, generated=None):
    """
    List {n} some {thing} names in the format:

    """
    return parse(generated, list_items)

def batch(tasks, ops):
    for n, thing in tasks:
        list_some(n, thing, **ops)
    return 1

def multi_generate(tasks, procs, **ops):
    tasks = [tasks[i::procs] for i in range(procs)]
    tasks = [(task, ops) for task in tasks]
    with mp.Pool(procs) as pool:
        pool.starmap(
             batch, tasks
        )

if __name__ == '__main__':

    tasks = [
        (2, 'animal'),
        (4, 'song'),
        (3, 'movie'),
        (9, 'car'),
        (6, 'city')
    ]
    
    list_some(5, 'tools', log=print)
    
    multi_generate(tasks, 3, log=print)

```




