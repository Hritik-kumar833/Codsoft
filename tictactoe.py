import copy

def display_board(game_board):
    for row in game_board:
        print(" | ".join(row))
        print("---------")

def has_winner(game_board, current_player):
    # Check rows, columns, and diagonals for a win
    for row in game_board:
        if all(cell == current_player for cell in row):
            return True
    for column in range(3):
        if all(game_board[row][column] == current_player for row in range(3)):
            return True
    if all(game_board[i][i] == current_player for i in range(3)) or all(game_board[i][2 - i] == current_player for i in range(3)):
        return True
    return False

def is_full_board(game_board):
    return all(cell != ' ' for row in game_board for cell in row)

def get_available_cells(game_board):
    return [(i, j) for i in range(3) for j in range(3) if game_board[i][j] == ' ']

def evaluate_board(game_board):
    if has_winner(game_board, 'X'):
        return -1
    elif has_winner(game_board, 'O'):
        return 1
    elif is_full_board(game_board):
        return 0
    else:
        return None

def minimax(game_board, depth, is_maximizing):
    board_score = evaluate_board(game_board)

    if board_score is not None:
        return board_score

    if is_maximizing:
        max_evaluation = float('-inf')
        for i, j in get_available_cells(game_board):
            game_board[i][j] = 'O'
            current_evaluation = minimax(game_board, depth + 1, False)
            game_board[i][j] = ' '
            max_evaluation = max(max_evaluation, current_evaluation)
        return max_evaluation
    else:
        min_evaluation = float('inf')
        for i, j in get_available_cells(game_board):
            game_board[i][j] = 'X'
            current_evaluation = minimax(game_board, depth + 1, True)
            game_board[i][j] = ' '
            min_evaluation = min(min_evaluation, current_evaluation)
        return min_evaluation

def determine_best_move(game_board):
    highest_value = float('-inf')
    optimal_move = None

    for i, j in get_available_cells(game_board):
        game_board[i][j] = 'O'
        move_value = minimax(game_board, 0, False)
        game_board[i][j] = ' '

        if move_value > highest_value:
            optimal_move = (i, j)
            highest_value = move_value

    return optimal_move

def execute_game():
    game_board = [[' ' for _ in range(3)] for _ in range(3)]
    is_player_turn = True  # True for 'X', False for 'O'

    while True:
        display_board(game_board)

        if is_player_turn:
            row = int(input("Enter row (0, 1, or 2): "))
            column = int(input("Enter column (0, 1, or 2): "))
            if game_board[row][column] == ' ':
                game_board[row][column] = 'X'
            else:
                print("Cell already taken. Try again.")
                continue
        else:
            print("AI is making a move...")
            optimal_move = determine_best_move(game_board)
            game_board[optimal_move[0]][optimal_move[1]] = 'O'

        if has_winner(game_board, 'X'):
            display_board(game_board)
            print("You win!")
            break
        elif has_winner(game_board, 'O'):
            display_board(game_board)
            print("AI wins!")
            break
        elif is_full_board(game_board):
            display_board(game_board)
            print("It's a tie!")
            break

        is_player_turn = not is_player_turn

if __name__ == "__main__":
    execute_game()
