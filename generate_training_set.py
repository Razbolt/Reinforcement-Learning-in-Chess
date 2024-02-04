#Train the model with given pgn data 
import chess.pgn
import os
from board import State
import numpy as np

import h5py


#Read the pgn files from the directory



def get_dataset(dir,num_samples = None):
    X,Y = [] , []
    gn = 0
    games = []
    for file in os.listdir(dir):
        if file.endswith(".pgn"):
            pgn = open(os.path.join(dir, file))
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception:
                    break

                games.append(game)
                
                result = game.headers["Result"]
                if game.headers['Result'] in ['1-0','0-1','1/2-1/2']:
                    value = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[game.headers["Result"]]
                else:
                    print("Skipping game %s, unknown result type" % result)
                    continue

                board = game.board()
               
                #print(value)
                
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    ser =State(board).serialize()[:,:,0]
                    #print(ser)
                    X.append(ser)
                    Y.append(value)
                print("parsing game %d,got %d examples " % (len(games),len(X)))

                if  num_samples is not None and len(X) > num_samples:
                    return X,Y
                gn += 1
    X = np.array(X)
    Y = np.array(Y)
    return X,Y           
                
            

if __name__ == "__main__":

    X,Y = get_dataset('data/',100000)
    np.savez("processed/dataset_100K.npz",X,Y)
