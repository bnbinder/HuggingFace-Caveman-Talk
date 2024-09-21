from rich.console import Console
from rich.live import Live
from time import sleep

console = Console()
data = [1, 2, 3, 4, 5]

with Live(f"Data: {data}", refresh_per_second=4) as live:
    for i in range(10):
        data.append(i)
        live.update(f"Data: {data}")
        sleep(1)






import click

from rich.console import Console
from rich.table import Table

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Production Budget", justify="right")
table.add_column("Box Office", justify="right")
table.add_row(
    "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)



from rich.table import Table

table = Table(title="Fruits Table")

table.add_column("Fruit", justify="right", style="cyan", no_wrap=True)
table.add_column("Color", style="magenta")
table.add_column("Price", justify="right", style="green")

table.add_row("Apple", "Red", "$1.00")
table.add_row("Banana", "Yellow", "$0.50")
table.add_row("Cherry", "Red", "$2.00")

console.print(table)





from rich.console import Console
from rich.traceback import install

install()  # This installs the Rich traceback handler

console = Console()

def divide(x, y):
    return x / y

try:
    result = divide(10, 0)
except Exception as e:
    console.print_exception(show_locals=True)
    
    
    
    
from rich.syntax import Syntax

code = """
def hello():
    print("Hello, World!")
"""

syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
console.print(syntax)




from rich.markdown import Markdown

markdown_text = """
# Hello World

This is a sample markdown text.

- Item 1
- Item 2
- Item 3
"""

md = Markdown(markdown_text)
console.print(md)



from rich.console import Console

console = Console()

console.print("Hello, [bold magenta]World![/bold magenta]")
console.print("[underline]This is underlined text.[/underline]")
console.print("[red]This text is red.[/red] and [green]this is green.[/green]")

import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="DEBUG", 
    format="%(message)s", 
    handlers=[RichHandler()]
)

log = logging.getLogger("rich")

log.debug("This is a debug message")
log.info("This is an info message")
log.warning("This is a warning")
log.error("This is an error")
log.critical("This is critical")

import time
from rich.progress import track

for i in track(range(10), description="Processing..."):
    time.sleep(0.5)
    
    
    
    
    
    
    
from time import sleep
from rich.console import Console

console = Console()
tasks = [f"task {n}" for n in range(1, 11)]

with console.status("[bold green]Working on tasks...") as status:
    while tasks:
        task = tasks.pop(0)
        sleep(1)
        console.log(f"{task} complete")
        

