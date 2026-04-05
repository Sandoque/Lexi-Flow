"""Ponto de entrada para executar o servidor Flask do LexiFlow."""

from app import create_app

app = create_app()


if __name__ == "__main__":
    app.run()
