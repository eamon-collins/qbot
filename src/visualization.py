import sys
sys.path.append('/home/eamon/repos/Quoridor-Online')
#viz = importlib.util.spec_from_file_location("visualize_gamestate", "../../Quoridor-Online/quoridor/client/__main__.py")
from quoridor.client.bot_integration import visualize_gamestate

def scatter(x, y, color):
	print("PYTHON RECEIVED")
	return 1


def main():
	visualize_gamestate([],[])


if __name__ == "__main__":
	main()