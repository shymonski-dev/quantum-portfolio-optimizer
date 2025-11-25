# quantum_portfolio_optimizer/src/quantum_portfolio_optimizer/cli.py
import click
from .utils import load_config
from .data import fetch_stock_data
from .simulation import get_provider
from .core import PortfolioVQESolver
import pprint

@click.group()
def cli():
    """
    Quantum Portfolio Optimizer CLI
    """
    pass

@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Path to the configuration file.')
def run(config):
    """
    Run a portfolio optimization experiment.
    """
    click.echo(f"Loading configuration from: {config}")
    try:
        config_data = load_config(config)
        click.echo("Configuration loaded successfully:")
        pprint.pprint(config_data)

        click.echo("\nFetching stock data...")
        portfolio_config = config_data.get('portfolio', {})
        tickers = portfolio_config.get('tickers')
        start_date = portfolio_config.get('start_date')
        end_date = portfolio_config.get('end_date')

        if not all([tickers, start_date, end_date]):
            raise ValueError("Tickers, start_date, and end_date must be defined in the portfolio config.")

        stock_data = fetch_stock_data(tickers, start_date, end_date)
        click.echo("Stock data fetched successfully:")
        click.echo(stock_data.head())

        click.echo("\nInitializing backend...")
        backend_config = config_data.get('backend', {})
        estimator, sampler = get_provider(backend_config)
        click.echo(f"Backend '{backend_config.get('name')}' initialized successfully.")
        click.echo(f"Estimator: {estimator}")
        click.echo(f"Sampler: {sampler}")

        click.echo("\nInitializing VQE solver...")
        algo_config = config_data.get('algorithm', {})
        optimizer_config = config_data.get('optimizer', {})
        
        solver = PortfolioVQESolver(
            estimator=estimator,
            ansatz_name=algo_config.get('settings', {}).get('ansatz'),
            optimizer_config=optimizer_config
        )
        click.echo("VQE solver initialized successfully.")

        # This is where the main optimization logic will be triggered.

    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        click.secho(str(e), fg='red')
    except Exception as e:
        click.secho(f"An error occurred: {e}", fg='red')

if __name__ == '__main__':
    cli()
