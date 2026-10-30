import sys

from app import register


def main(argv):
    name, email = argv[1], argv[2]
    print(register(name, email))


if __name__ == "__main__":
    main(sys.argv)
