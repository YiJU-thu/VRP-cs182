import os

def clear_concorde_files():
    for fn in os.listdir():
        # end with .pul, .sav, .res, .tsp
        if fn.endswith(".sav") or fn.endswith(".pul")\
             or fn.endswith(".res") or fn.endswith(".tsp") or fn.endswith(".sol"):
            os.remove(fn)
        # end with *.[0-9][0-9][0-9]
        elif fn[-3:].isdigit() and fn[-4] == ".":
            os.remove(fn)

if __name__ == "__main__":
    clear_concorde_files()