"""
Recursive Backtracking Solver w/ Forward-Checking for Kropki Sudoku Puzzles
Author: Navaneeth Rajan
"""
import argparse


# Command-line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Solve a Kropki Sudoku puzzle.")
    parser.add_argument("input_file", help="Path to the input file containing the puzzle.")
    parser.add_argument(
        "output_file",
        nargs="?",
        default="Solved_Output.txt",
        help="Path to the output file where the solved puzzle will be saved (default: Solved_Output.txt)."
    )
    return parser.parse_args()


# Initializes legal possibilities grid based on initial board state and dot locations
def initialize_possibilities(board, hdots, vdots):

    # Generate possibility set
    possibilities = [[set() for i in range(9)] for i in range(9)]

    for row in range(9):
        for col in range(9):
            # If slot is empty, put in all legal possibilities
            if board[row][col] == 0:
                possibilities[row][col] = set(range(1, 10))
            # If slot is full, nothing to put in
            else:
                possibilities[row][col] = set()

    # Eliminate invalid possibilities based on the current board state
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                toRemove = []
                for choice in possibilities[row][col]:
                    valid = is_valid(board, row, col, choice, hdots, vdots)
                    if not valid:
                        toRemove.append(choice)
                for choice in toRemove:
                    possibilities[row][col].remove(choice)

    return possibilities


# Returns the degree heuristic of a tile, based on current board state and dot locations
def degree_heuristic(row, col, board, hdots, vdots):
    degree = 0

    # Count row and column constraints
    for i in range(9):
        if i != col and board[row][i] == 0:  # Unassigned tile in the same row
            degree += 1
        if i != row and board[i][col] == 0:  # Unassigned tile in the same column
            degree += 1

    # Count box constraints
    box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            r, c = box_start_row + i, box_start_col + j

            # We don't want to double count anything already checked in the row or col counts
            if r == row or c == col: continue
            if (r, c) != (row, col) and board[r][c] == 0:  # Unassigned tile in the box
                degree += 1

    # Count horizontal Kropki dot constraints
    # Note all these tiles would've been counted as part of the row count
    # However, we're intentionally double-counting dot nodes, since they're especially constrained
    if col > 0 and hdots[row][col - 1] != 0 and board[row][col - 1] == 0:  # Left neighbor is unassigned
        degree += 1
    if col < 8 and hdots[row][col] != 0 and board[row][col + 1] == 0:  # Right neighbor is unassigned
        degree += 1

    # Count vertical Kropki dot constraints
    # Note all these tiles would've been counted as part of the column count
    # However, we're intentionally double-counting dot nodes, since they're especially constrained
    if row > 0 and vdots[row - 1][col] != 0 and board[row - 1][col] == 0:  # Above neighbor is unassigned
        degree += 1
    if row < 8 and vdots[row][col] != 0 and board[row + 1][col] == 0:  # Below neighbor is unassigned
        degree += 1

    return degree


# Forward checks, seeing if a selection would cause an empty tile to have no legal possibilities.
# If the check passes, returns a dictionary with the possibilities that should be removed given the choice.
def fwd_check_and_update_possibilities(possibilities, row, col, num, board, hdots, vdots, ):
    def apply_kropki_constraints(possibilities, removals, row, col, num, dot_type):
        if (row, col) not in removals:
            removals[(row, col)] = set()

        if dot_type == 1:  # White dot (consecutive numbers)
            for val in possibilities[row][col]:
                if val != (num - 1) and val != (num + 1):
                    removals[(row, col)].add(val)

        elif dot_type == 2:  # Black dot (2:1 ratio)
            for val in possibilities[row][col]:
                if val != (num * 2) and val != (num / 2):
                    removals[(row, col)].add(val)

    removals = {(row, col): set()}
    for i in possibilities[row][col]:
        removals[(row, col)].add(i)

    for i in range(9):
        if board[row][i] == 0 and num in possibilities[row][i]:  # Unassigned tile in the same row
            removals[(row, i)] = set()
            removals[(row, i)].add(num)
        if board[i][col] == 0 and num in possibilities[i][col]:  # Unassigned tile in the same column
            removals[(i, col)] = set()
            removals[(i, col)].add(num)

    box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            r, c = box_start_row + i, box_start_col + j
            if board[r][c] == 0 and num in possibilities[r][c]:
                removals[(r, c)] = set()
                removals[(r, c)].add(num)

    if col > 0 and hdots[row][col - 1] != 0:  # Left neighbor
        if board[row][col - 1] == 0:
            apply_kropki_constraints(possibilities, removals, row, col - 1, num, hdots[row][col - 1])

    if col < 8 and hdots[row][col] != 0:  # Right neighbor
        if board[row][col + 1] == 0:
            apply_kropki_constraints(possibilities, removals, row, col + 1, num, hdots[row][col])

        # Update vertical Kropki dot constraints
    if row > 0 and vdots[row - 1][col] != 0:  # Above neighbor
        if board[row - 1][col] == 0:
            apply_kropki_constraints(possibilities, removals, row - 1, col, num, vdots[row - 1][col])

    if row < 8 and vdots[row][col] != 0:  # Below neighbor
        if board[row + 1][col] == 0:
            apply_kropki_constraints(possibilities, removals, row + 1, col, num, vdots[row][col])

    for location in removals:

        r = location[0]
        c = location[1]

        if (r, c) != (row, col) and len(removals[location]) == len(possibilities[r][c]):
            return False

    return removals


# Prints a passed board state. Doesn't print dots.
def print_board(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                print(board[i][j], end="  ")
            else:
                print("_", end="  ")
        print("")


# Selects the best empty location to recurse on
# Uses minimum values heuristic, with degree heuristic as a tiebreaker

def next_spot(board, hdots, vdots):
    unassigned_spots = set()

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                unassigned_spots.add((row, col))

    leader = None
    least_possibilities = 10
    highest_degree = -1

    for spot in unassigned_spots:
        r, c = spot[0], spot[1]
        dh = degree_heuristic(r, c, board, hdots, vdots)

        if len(possibilities[r][c]) < least_possibilities:
            leader = spot
            least_possibilities = len(possibilities[r][c])
            highest_degree = dh

        elif len(possibilities[r][c]) == least_possibilities and dh > highest_degree:
            leader = spot
            least_possibilities = len(possibilities[r][c])
            highest_degree = dh

    return leader

# Returns True if a specific entry for a tile would be legal, False otherwise
# Used to initialize possibility sets
def is_valid(board, row, col, num, hdots, vdots):

    # Is the choice unused in the row?
    def valid_row(arr, row, num):
        for i in range(9):
            if (arr[row][i] == num):
                return False
        return True

    # Is the choice unused in the column?
    def valid_col(arr, col, num):
        for i in range(9):
            if (arr[i][col] == num):
                return False
        return True

    # Is the choice unused in the box?
    def valid_box(arr, row, col, num):
        for i in range(3):
            for j in range(3):
                if (arr[i + row][j + col] == num):
                    return False
        return True

    # Does this choice satisfy horizontal Kropki dot constraints?
    def valid_hdots(board, hdots, row, col, num):
        if col > 0 and hdots[row][col - 1] != 0:
            left = board[row][col - 1]
            if left != 0:
                if hdots[row][col - 1] == 1 and abs(left - num) != 1:
                    return False
                if hdots[row][col - 1] == 2 and (left != 2 * num and num != 2 * left):
                    return False
        if col < 8 and hdots[row][col] != 0:
            right = board[row][col + 1]
            if right != 0:
                if hdots[row][col] == 1 and abs(right - num) != 1:
                    return False
                if hdots[row][col] == 2 and (right != 2 * num and num != 2 * right):
                    return False
        return True

    # Does this choice satisfy vertical Kropki dot constraints?
    def valid_vdots(board, vdots, row, col, num):
        if row > 0 and vdots[row - 1][col] != 0:
            above = board[row - 1][col]
            if above != 0:
                if vdots[row - 1][col] == 1 and abs(above - num) != 1:
                    return False
                if vdots[row - 1][col] == 2 and (above != 2 * num and num != 2 * above):
                    return False
        if row < 8 and vdots[row][col] != 0:
            below = board[row + 1][col]
            if below != 0:
                if vdots[row][col] == 1 and abs(below - num) != 1:
                    return False
                if vdots[row][col] == 2 and (below != 2 * num and num != 2 * below):
                    return False
        return True

    row_valid = valid_row(board, row, num)
    col_valid = valid_col(board, col, num)
    box_valid = valid_box(board, row - row % 3, col - col % 3, num)
    hdots_valid = valid_hdots(board, hdots, row, col, num)
    vdots_valid = valid_vdots(board, vdots, row, col, num)

    return row_valid and col_valid and box_valid and hdots_valid and vdots_valid


def kropki_solve(board, hdots, vdots, possibilities):
    empty_loc = next_spot(board, hdots, vdots)
    if empty_loc is None:
        return True

    row = empty_loc[0]
    col = empty_loc[1]

    choice_list = list(possibilities[row][col])
    for choice in choice_list:

        removals = fwd_check_and_update_possibilities(possibilities, row, col, choice, board, hdots, vdots)

        # If forward checking fails, move on: this possibility isn't going to work
        if removals is False:
            continue

        # Otherwise, update assumptions based on this selection
        for location in removals:
            for removed in removals[location]:
                r = location[0]
                c = location[1]
                possibilities[r][c].remove(removed)

        # Update the grid based on the choice
        board[row][col] = choice

        # Recursively solve the grid based on the choice
        if kropki_solve(board, hdots, vdots, possibilities):
            return True

        # Undo your grid change
        board[row][col] = 0

        # Remove any now-invalid assumptions
        for location in removals:
            for removed in removals[location]:
                r = location[0]
                c = location[1]
                possibilities[r][c].add(removed)

    return False





if __name__ == "__main__":

    args = parse_arguments()
    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, 'r') as f:
        lines = f.read().strip().split('\n')

    # creating a 2D array for the grid
    board = [[0 for x in range(9)] for y in range(9)]
    for i in range(9):
        row_values = list(map(int, lines[i].split()))
        board[i] = row_values

    # Parse horizontal dots into a 2D list
    horizontal_dots = [[0 for x in range(8)] for y in range(9)]
    for i in range(9):
        row_values = list(map(int, lines[10 + i].split()))
        horizontal_dots[i] = row_values

    # Parse vertical dots into a 2D list
    vertical_dots = [[0 for x in range(9)] for y in range(8)]
    for i in range(8):
        row_values = list(map(int, lines[20 + i].split()))
        vertical_dots[i] = row_values

    print("Initial:")
    print_board(board)
    print("\n")

    possibilities = initialize_possibilities(board, horizontal_dots, vertical_dots)

    # if successful, print the grid
    if kropki_solve(board, horizontal_dots, vertical_dots, possibilities):
        print("Solution:")
        print_board(board)

        with open(output_file, 'w') as file:
            for i in range(len(board)):
                row = board[i]
                if i < 8:
                    file.write(' '.join(map(str, row)) + '\n')
                else:
                    file.write(' '.join(map(str, row)))
    else:
        print("No solution found")