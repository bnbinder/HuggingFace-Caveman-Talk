import click

@click.group()
def cli():
    """A simple CLI tool."""
    pass

@cli.command()
@click.argument('name')
@click.option('--greeting', default='Hello', help='Greeting to use.')
def greet(name, greeting):
    """Greet a person."""
    click.echo(f"{greeting}, {name}!")

@cli.command()
@click.argument('number', type=int)
def square(number):
    """Print the square of a number."""
    click.echo(f"The square of {number} is {number ** 2}")

if __name__ == '__main__':
    cli()
