from rich.console import Console
from rich.panel import Panel

panel = Panel("", 
              title="Panel Title", subtitle="Panel Subtitle", expand=False)
console = Console()
console.print(panel)



from rich.console import Console
from datetime import datetime



print("hello")
print("hello", end='')
print("hello")