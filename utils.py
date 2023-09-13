def format_state(state, multiLine=False):
    # Format the state as a string for printing
    row_sp = multiLine and "\n" or " | "
    col_sp = multiLine and "\t" or " "
    return row_sp.join(col_sp.join(map(str, row)) for row in state)


ACTION_NAMES = "↑↓←→"