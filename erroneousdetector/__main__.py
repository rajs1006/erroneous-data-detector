import sys

import streamlit.cli as stcli

from erroneousdetector.main import Main
from erroneousdetector.main.utils.Constants import Environment as envCons
from erroneousdetector.main.utils.Constants import args


def main():

    if args.action == "gui":
        sys.argv = [
            "streamlit",
            "run",
            "{}/main/Dashboard.py".format(envCons.app),
            "--",
            "--env={}".format(args.env),
        ]
        sys.exit(stcli.main())
    else:
        Main.main()


if __name__ == "__main__":
    main()
